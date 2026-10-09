"""CNN14 must see the same fbank as speechbrain's classify_batch.

Only clips under Cnn14's 32-frame minimum (where classify_batch crashes)
may be padded. The front end is speechbrain's, the embedding model a stub.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from mteb._create_dataloaders import _custom_collate_fn
from mteb.models.model_implementations.cnn14_model import CNN14Wrapper

features = pytest.importorskip("speechbrain.processing.features")

SAMPLING_RATE = 44_100


class _AudioDataset(Dataset):
    features = {"audio": None}

    def __init__(self, clips: list[np.ndarray]) -> None:
        self.clips = clips

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, i: int) -> dict:
        return {"audio": {"array": self.clips[i], "sampling_rate": SAMPLING_RATE}}


@pytest.mark.parametrize(
    ("seconds", "n_frames"),
    [
        (0.2, 32),  # 18 frames: padded to the minimum
        (0.6, 52),  # classify_batch takes it unpadded
        (2.0, 173),
    ],
)
def test_fbank_padded_only_below_minimum(seconds, n_frames) -> None:
    # speechbrain/cnn14-esc50 hyperparams.yaml
    compute_stft = features.STFT(
        sample_rate=SAMPLING_RATE, n_fft=1024, win_length=23.2199, hop_length=11.6099
    )
    compute_fbank = features.Filterbank(
        n_mels=80, n_fft=1024, sample_rate=SAMPLING_RATE
    )
    seen = []

    def embedding_model(feats: torch.Tensor) -> torch.Tensor:
        seen.append(feats)
        return feats.mean(dim=1, keepdim=True)

    wrapper = CNN14Wrapper.__new__(CNN14Wrapper)
    wrapper.device = "cpu"
    wrapper.max_audio_length_seconds = None
    wrapper.sampling_rate = SAMPLING_RATE
    wrapper.model = SimpleNamespace(
        mods=SimpleNamespace(
            compute_stft=compute_stft,
            compute_fbank=compute_fbank,
            embedding_model=embedding_model,
        ),
        hparams=SimpleNamespace(spec_mag_power=0.5),
    )
    rng = np.random.default_rng(0)
    clip = (rng.standard_normal(int(seconds * SAMPLING_RATE)) * 0.1).astype(np.float32)
    loader = DataLoader(
        _AudioDataset([clip]), batch_size=1, collate_fn=_custom_collate_fn
    )

    wrapper.get_audio_embeddings(loader, show_progress_bar=False)

    with torch.no_grad():
        stft = compute_stft(torch.from_numpy(clip)[None])
        expected = compute_fbank(features.spectral_magnitude(stft, power=0.5))
    assert seen[0].shape == (1, n_frames, 80)
    torch.testing.assert_close(seen[0][:, : expected.shape[1]], expected)
    assert not seen[0][:, expected.shape[1] :].any()
