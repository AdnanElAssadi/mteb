from __future__ import annotations

import logging
import math
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
from tqdm.auto import tqdm

from mteb.models import ModelMeta
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.modality_collators import seconds_to_samples

if TYPE_CHECKING:
    import torch
    from torch.utils.data import DataLoader

    from mteb import TaskMetadata
    from mteb.models.models_protocols import EncoderProtocol
    from mteb.types import Array, BatchedInput, PromptType
    from mteb.types._encoder_io import AudioInput


logger = logging.getLogger(__name__)


def yamnet_loader(*args: Any, **kwargs: Any) -> EncoderProtocol:
    """Factory function to create a YAMNet model wrapper."""
    import torch
    from torch_vggish_yamnet import yamnet

    class YAMNetWrapper(AbsEncoder):
        def __init__(
            self,
            device: str | None = None,
            # no limit: 0.96 s patches, hop 0.48, no max
            # https://github.com/tensorflow/models/blob/master/research/audioset/yamnet/params.py
            max_audio_length_seconds: float | None = None,
            **kwargs: Any,
        ):
            import torch

            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"

            self.device = device
            self.max_audio_length_seconds = max_audio_length_seconds
            self.model = yamnet.yamnet(pretrained=True)
            self.model.eval().to(self.device)

            self.sampling_rate = 16000  # YAMNet requires 16kHz audio
            # official log-mel front end (yamnet/features.py): the port's
            # torchaudio MelSpectrogram centres (reflect-pads) its STFT frames
            # and builds its mel filterbank slightly differently
            self.stft_window = torch.hann_window(400, periodic=True)  # 25 ms
            self.stft_hop = 160  # 10 ms
            self.fft_length = 512
            self.mel_matrix = torch.from_numpy(_linear_to_mel_weight_matrix()).float()
            self.log_offset = 0.001
            self.embed_dim = 1024  # YAMNet embedding dimension
            # official patching: 0.96 s patches with a 0.48 s hop, the waveform
            # padded to one full patch and then to a whole number of hops
            # https://github.com/tensorflow/models/blob/master/research/audioset/yamnet/features.py
            self.patch_frames = 96  # 0.96 s of 10 ms frames
            self.patch_hop_frames = 48  # 0.48 s
            # one patch plus the STFT window minus one hop: 0.975 s
            self.min_samples = round(0.975 * self.sampling_rate)  # 15,600
            self.hop_samples = round(0.48 * self.sampling_rate)  # 7,680

        def _resample_audio(
            self, audio: torch.Tensor, source_rate: int
        ) -> torch.Tensor:
            """Resample audio to target sampling rate."""
            import torchaudio

            if source_rate != self.sampling_rate:
                resampler = torchaudio.transforms.Resample(
                    source_rate, self.sampling_rate
                )
                return resampler(audio)
            return audio

        def _normalize_audio(self, audio: Array) -> torch.Tensor:
            """Normalize and pad audio to minimum length."""
            if isinstance(audio, np.ndarray):
                audio = torch.from_numpy(audio)

            audio = audio.float().squeeze()
            if audio.ndim > 1:
                audio = audio.mean(dim=0)

            # Apply audio truncation
            max_length = seconds_to_samples(
                self.max_audio_length_seconds, self.sampling_rate
            )
            if max_length is not None and audio.shape[-1] > max_length:
                audio = audio[..., :max_length]

            # Normalize to [-1.0, 1.0]
            if audio.numel() > 0 and audio.abs().max() > 1.0:
                audio = audio / audio.abs().max()  # noqa: PLR6104

            # Pad to one patch, then fill out the last hop (pad_waveform)
            num_samples = max(audio.shape[-1], self.min_samples)
            num_hops = math.ceil((num_samples - self.min_samples) / self.hop_samples)
            target = self.min_samples + num_hops * self.hop_samples
            audio = torch.nn.functional.pad(audio, (0, target - audio.shape[-1]))

            return audio

        def _log_mel(self, audio: torch.Tensor) -> torch.Tensor:
            """Log-mel spectrogram [frames, mels] as tf.signal.stft (no centring)."""
            frames = audio.unfold(-1, len(self.stft_window), self.stft_hop)
            magnitude = torch.fft.rfft(frames * self.stft_window, n=self.fft_length)
            mel = magnitude.abs() @ self.mel_matrix
            return torch.log(mel + self.log_offset)

        def _prepare_input_tensor(self, audio_data: Array) -> torch.Tensor | None:
            """Convert audio to YAMNet input format and handle tensor dimensions."""
            if isinstance(audio_data, np.ndarray):
                audio_data = torch.from_numpy(audio_data)

            if audio_data.ndim == 1:
                audio_data = audio_data.unsqueeze(0)

            # [frames, mels]; the port's own WaveformToInput cuts patches with a
            # 0.96 s hop, so cut them here with 0.48 s
            log_mel = self._log_mel(audio_data.float().squeeze(0))
            patches = log_mel.unfold(0, self.patch_frames, self.patch_hop_frames)
            # [patches, mels, frames] -> [patches, 1, frames, mels]
            input_tensor = patches.transpose(1, 2).unsqueeze(1).to(self.device)

            if input_tensor.numel() == 0:
                return None

            return input_tensor

        def get_audio_embeddings(
            self,
            inputs: DataLoader[AudioInput],
            show_progress_bar: bool = True,
            **kwargs: Any,
        ) -> Array:
            """Generate embeddings for audio inputs."""
            all_embeddings = []

            for batch in tqdm(
                inputs,
                desc="Processing audio",
                disable=not show_progress_bar,
            ):
                batch_embeddings = []

                for a in batch["audio"]:
                    array = torch.tensor(a["array"], dtype=torch.float32)
                    sr = a["sampling_rate"]
                    if sr is None:
                        warnings.warn(
                            f"No sampling_rate provided for an audio sample. "
                            f"Assuming {self.sampling_rate} Hz (model default).",
                            stacklevel=2,
                        )
                        sr = self.sampling_rate

                    # Resample if needed
                    audio = self._resample_audio(array.float(), sr)
                    # Normalize
                    audio = self._normalize_audio(audio)

                    with torch.no_grad():
                        input_tensor = self._prepare_input_tensor(audio)
                        # discard logits
                        embedding, _ = self.model(input_tensor)

                        # Handle 4D tensors with shape [batch, embed_dim, 1, 1]
                        if (
                            embedding.dim() == 4
                            and embedding.shape[2] == 1
                            and embedding.shape[3] == 1
                        ):
                            # [batch, embed_dim, 1, 1] -> [batch, embed_dim]
                            embedding = embedding.squeeze(2).squeeze(2)

                        # Use mean pooling if needed
                        if len(embedding.shape) > 1:
                            embedding = torch.mean(embedding, dim=0)

                        batch_embeddings.append(embedding.cpu().detach().numpy())

                all_embeddings.extend(batch_embeddings)

            return np.array(all_embeddings)

        def encode(
            self,
            inputs: DataLoader[BatchedInput],
            *,
            task_metadata: TaskMetadata,
            hf_split: str,
            hf_subset: str,
            prompt_type: PromptType | None = None,
            **kwargs: Any,
        ) -> Array:
            if "audio" not in inputs.dataset.features:
                raise ValueError("YAMNetWrapper only supports audio inputs.")
            return self.get_audio_embeddings(inputs, **kwargs)

    return YAMNetWrapper(**kwargs)


def _linear_to_mel_weight_matrix(
    num_mel_bins: int = 64,
    num_spectrogram_bins: int = 257,
    sample_rate: float = 16000.0,
    lower_edge_hertz: float = 125.0,
    upper_edge_hertz: float = 7500.0,
) -> np.ndarray:
    """NumPy port of tf.signal.linear_to_mel_weight_matrix (HTK mel, DC bin zeroed)."""

    def hertz_to_mel(hertz: np.ndarray) -> np.ndarray:
        return 1127.0 * np.log1p(hertz / 700.0)

    linear_frequencies = np.linspace(0.0, sample_rate / 2.0, num_spectrogram_bins)[1:]
    spectrogram_bins_mel = hertz_to_mel(linear_frequencies)[:, None]
    band_edges_mel = np.linspace(
        hertz_to_mel(lower_edge_hertz),
        hertz_to_mel(upper_edge_hertz),
        num_mel_bins + 2,
    )
    lower_edge_mel = band_edges_mel[:-2]
    center_mel = band_edges_mel[1:-1]
    upper_edge_mel = band_edges_mel[2:]
    lower_slopes = (spectrogram_bins_mel - lower_edge_mel) / (
        center_mel - lower_edge_mel
    )
    upper_slopes = (upper_edge_mel - spectrogram_bins_mel) / (
        upper_edge_mel - center_mel
    )
    mel_weights = np.maximum(0.0, np.minimum(lower_slopes, upper_slopes))
    return np.pad(mel_weights, [[1, 0], [0, 0]])


yamnet = ModelMeta(
    loader=yamnet_loader,
    name="google/yamnet",
    languages=["eng-Latn"],
    open_weights=True,
    revision="1",
    release_date="2020-10-06",
    max_tokens=float("inf"),
    n_parameters=3_751_369,
    n_embedding_parameters=0,
    memory_usage_mb=14,
    embed_dim=1024,
    license="apache-2.0",
    reference="https://tfhub.dev/google/yamnet/1",
    similarity_fn_name="cosine",
    framework=["TensorFlow"],
    use_instructions=False,
    public_training_code="https://github.com/tensorflow/models/tree/master/research/audioset/yamnet",
    public_training_data="https://research.google.com/audioset/",
    training_datasets={
        "AudioSetMini",
    },
    modalities=["audio"],
    citation="""
@inproceedings{audioset,
  title={Audio set: An ontology and human-labeled dataset for audio events},
  author={Gemmeke, Jort F and Ellis, Daniel PW and Freedman, Dylan and Jansen, Aren and Lawrence, Wade and Moore, R Channing and Plakal, Manoj and Ritter, Marvin},
  booktitle={2017 IEEE international conference on acoustics, speech and signal processing (ICASSP)},
  pages={776--780},
  year={2017},
  organization={IEEE}
}
""",
    extra_requirements_groups=["torch-vggish-yamnet"],
)
