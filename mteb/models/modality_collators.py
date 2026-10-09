from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Literal, TypedDict, cast

import numpy as np

from mteb._log_once import LogOnce
from mteb.types._encoder_io import AudioInputItem

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    import torch
    from torch.utils.data import DataLoader
    from torchcodec.decoders import VideoDecoder  # type: ignore[attr-defined]

    from mteb.types import BatchedInput

logger = logging.getLogger(__name__)
log_once = LogOnce(logger)

# FPS-based sampling follows qwen-vl-utils (FPS_MIN_FRAMES, FRAME_FACTOR), which
# almost every fps-sampled model here was trained with. Qwen-VL video
# processors need at least 2 frames (`t:1 must be larger than temporal_factor:2`).
DEFAULT_MIN_FRAMES = 4
DEFAULT_FRAME_FACTOR = 2


def frame_indices(n_source: int, target: int) -> list[int]:
    """Indices of ``target`` evenly spaced frames from a clip of ``n_source`` frames."""
    if n_source < target:
        # repeat each frame in place (0, 0, 1, 1, ...) instead of looping
        # back to the start, as linspace-based upstream samplers do
        return [i * n_source // target for i in range(target)]
    return [int(i) for i in np.linspace(0, n_source - 1, target).round()]


class VideoFrameMetadata(TypedDict):
    """Where the sampled frames of a clip come from.

    The keys match ``transformers.video_utils.VideoMetadata``, so the dict can be
    passed to a video processor as ``video_metadata`` (or as
    ``{"array": frames, "video_metadata": ...}`` to sentence-transformers).

    Attributes:
        total_num_frames: Number of (decodable) frames in the source clip.
        fps: Frame rate of the source clip.
        duration: Duration of the source clip in seconds.
        frames_indices: Source index of each sampled frame.
    """

    total_num_frames: int
    fps: float | None
    duration: float | None
    frames_indices: list[int]


def sampled_fps(metadata: VideoFrameMetadata) -> float | None:
    """Rate of the sampled frames, as qwen-vl-utils reports it to the processor.

    ``len(frames_indices) / total_num_frames * fps``; None without a source fps.
    """
    if not metadata["fps"] or not metadata["total_num_frames"]:
        return None
    return (
        len(metadata["frames_indices"]) / metadata["total_num_frames"] * metadata["fps"]
    )


def fps_shortfall(
    metadata: VideoFrameMetadata,
    fps: float,
    frame_factor: int = DEFAULT_FRAME_FACTOR,
) -> float:
    """How much slower than ``fps`` a clip was sampled, as ``fps / sampled_fps``.

    1.0 when the clip got the frames FPS-based sampling asks for (up to the
    rounding of the frame count). Above 1.0 when ``max_frames`` capped a long
    clip or the source has fewer frames than ``fps`` asks for. Processors that
    assume a fixed frame rate (Qwen-Omni) need their frame timing stretched by
    this factor.
    """
    actual = sampled_fps(metadata)
    if actual is None:
        return 1.0
    n_frames = len(metadata["frames_indices"])
    duration = n_frames / actual
    if n_frames + frame_factor >= duration * fps:
        return 1.0
    return fps / actual


AudioResampling = Literal["kaiser_best", "torchaudio_default"]
# torchaudio resampler settings per `resampling` choice. "kaiser_best" are
# resampy's kaiser_best parameters (librosa's former default) and stay close to
# librosa's soxr_hq; torchaudio's default (Hann window, width 6, rolloff 0.99)
# lets much of the band above the new Nyquist alias back. Use
# "torchaudio_default" for models whose authors resample with it.
_RESAMPLING_KWARGS: dict[str, dict[str, Any]] = {
    "kaiser_best": {
        "lowpass_filter_width": 64,
        "rolloff": 0.9475,
        "resampling_method": "sinc_interp_kaiser",
        "beta": 14.769656459379492,
    },
    "torchaudio_default": {},
}


def stretch_second_per_grid(
    second_per_grid: torch.Tensor,
    metadata: Sequence[VideoFrameMetadata | None],
    fps: float,
) -> torch.Tensor:
    """Correct Qwen-Omni ``video_second_per_grid`` for clips sampled below ``fps``.

    Qwen-Omni processors time every clip by one fixed frame rate. This keeps
    their value for clips that got the frames ``fps`` asks for (as the
    authors' pipelines do) and stretches it by `fps_shortfall` for clips
    capped by ``max_frames`` or from a slower source, so the temporal
    positions span the whole clip. Returns float32: bf16 would round e.g.
    2.81 to 2.8125.
    """
    import torch

    scale = torch.tensor(
        [fps_shortfall(m, fps) if m is not None else 1.0 for m in metadata],
        dtype=torch.float32,
        device=second_per_grid.device,
    )
    return second_per_grid.float() * scale


class AudioCollator:
    """Collator for audio data that resamples audio to a target sampling rate and optionally truncates to a maximum number of samples."""

    def __init__(
        self,
        *,
        target_sampling_rate: int,
        max_samples: int | None = None,
        resampling: AudioResampling = "kaiser_best",
    ) -> None:
        """Initialize the collator.

        Args:
            target_sampling_rate: The sampling rate to resample the audio to.
            max_samples: The maximum number of samples to keep for each audio. If None, no truncation is applied.
            resampling: Resampling filter, see `_RESAMPLING_KWARGS`.
        """
        self.target_sampling_rate = target_sampling_rate
        self.max_samples = max_samples
        self.resampling = resampling

    def __call__(self, inputs: list[dict[str, Any]]) -> BatchedInput:
        """Collate a batch of audio inputs by resampling them to the target sampling rate and optionally truncating to a maximum number of samples."""
        return self.resample_audios(
            inputs,
            target_sampling_rate=self.target_sampling_rate,
            max_samples=self.max_samples,
            resampling=self.resampling,
        )

    @staticmethod
    def resample_audios(
        inputs: list[dict[str, Any]],
        *,
        target_sampling_rate: int,
        max_samples: int | None = None,
        resampling: AudioResampling = "kaiser_best",
    ) -> BatchedInput:
        """Resample a batch of audio inputs to a target sampling rate and optionally truncate to a maximum number of samples.

        Args:
            inputs: A list of dictionaries containing audio data under the "audio" key, where each audio is a dictionary with "array" and "sampling_rate" keys.
            target_sampling_rate: The sampling rate to resample the audio to.
            max_samples: The maximum number of samples to keep for each audio. If None, no truncation is applied.
            resampling: Resampling filter, see `_RESAMPLING_KWARGS`.
        """
        collated_inputs = []
        for row in inputs:
            audio_array = AudioCollator.resample_audio(
                row,
                target_sampling_rate=target_sampling_rate,
                max_samples=max_samples,
                resampling=resampling,
            )
            row["audio"] = AudioInputItem(
                array=audio_array, sampling_rate=target_sampling_rate
            )
            collated_inputs.append(row)
        return cast(
            "BatchedInput",
            {key: [row[key] for row in collated_inputs] for key in collated_inputs[0]},
        )

    @staticmethod
    def resample_audio(
        audio: dict[str, Any],
        *,
        target_sampling_rate: int,
        max_samples: int | None = None,
        resampling: AudioResampling = "kaiser_best",
    ) -> np.typing.NDArray[np.floating]:
        """Resample an audio input to a target sampling rate and optionally truncate to a maximum number of samples.

        Args:
            audio: A list of dictionaries containing audio data under the "audio" key, where each audio is a dictionary with "array" and "sampling_rate" keys.
            target_sampling_rate: The sampling rate to resample the audio to.
            max_samples: The maximum number of samples to keep for each audio. If None, no truncation is applied.
            resampling: Resampling filter, see `_RESAMPLING_KWARGS`.
        """
        if max_samples is not None and max_samples <= 0:
            raise ValueError(f"max_samples must be positive, got {max_samples}.")
        import torch
        import torchaudio

        audio = audio["audio"]
        if audio["sampling_rate"] != target_sampling_rate:
            logger.debug(
                f"Resampling audio from {audio['sampling_rate']} Hz to {target_sampling_rate} Hz."
            )
            resampler = torchaudio.transforms.Resample(
                orig_freq=audio["sampling_rate"],
                new_freq=target_sampling_rate,
                **_RESAMPLING_KWARGS[resampling],
            )
            audio_array = resampler(
                torch.from_numpy(np.asarray(audio["array"])).float()
            ).numpy()
        else:
            audio_array = np.asarray(audio["array"])

        # Convert to mono if needed
        if audio_array.ndim > 1 and audio_array.shape[0] > 1:
            audio_array = np.mean(audio_array, axis=0)

        if max_samples is not None:
            num_samples = audio_array.shape[-1]
            if num_samples > max_samples:
                log_once.info(
                    f"Truncating audio longer than {max_samples / target_sampling_rate:g} s "
                    "(the configured audio cap); only the start of each longer clip is encoded."
                )
                audio_array = audio_array[..., :max_samples]

        audio_array = cast(
            "np.ndarray[tuple[Any, ...], np.dtype[np.floating[Any]]]",
            audio_array,
        )
        return audio_array


def check_duration_cap(seconds: float | None) -> float | None:
    """Validate an optional audio length cap (seconds or samples). None means no cap."""
    if seconds is not None and seconds <= 0:
        raise ValueError(f"Audio length cap must be positive, got {seconds}.")
    return seconds


def seconds_to_samples(seconds: float | None, sampling_rate: int) -> int | None:
    """Convert an optional duration cap to samples. None means no cap."""
    seconds = check_duration_cap(seconds)
    if seconds is None:
        return None
    return int(seconds * sampling_rate)


@contextmanager
def fixed_numpy_seed(seed: int = 0) -> Iterator[None]:
    """Seed numpy's global RNG for a block, then restore it.

    Some feature extractors pick random crops of long clips from the global
    RNG (CLAP fusion, LanguageBind audio); seeding per clip makes the
    embeddings reproducible.
    """
    state = np.random.get_state()  # noqa: NPY002
    np.random.seed(seed)  # noqa: NPY002
    try:
        yield
    finally:
        np.random.set_state(state)  # noqa: NPY002


def single_clip_dataloader(
    inputs: DataLoader[BatchedInput],
    reason: str,
) -> DataLoader[BatchedInput]:
    """Rebuild an audio or video DataLoader with batch_size=1.

    For models whose embeddings change with their batch-mates (batched vs single
    cosine below 0.99 on real clips). ``reason`` is logged once.
    ``DataLoader.batch_size`` cannot be changed after creation.
    """
    from torch.utils.data import DataLoader

    log_once.info(f"Encoding one item at a time (batch_size ignored): {reason}.")

    return DataLoader(
        inputs.dataset,
        batch_size=1,
        collate_fn=inputs.collate_fn,
        num_workers=inputs.num_workers,
    )


class FramesCollator:
    """Collator for video data that resamples video frames.

    Supports two sampling modes:
    - FPS-based: downsamples to ``fps`` frames per second, so the number of
      selected frames scales with video duration. Videos already below the
      target rate keep all frames. ``max_frames`` caps the total to prevent
      OOM on very long videos. As in qwen-vl-utils, at least ``min_frames``
      frames are taken and the count is rounded down to a multiple of
      ``frame_factor``.
    - Fixed-sample: always selects exactly ``num_frames`` frames uniformly
      across the video, regardless of duration. Clips with fewer frames
      repeat each frame in place.

    When both ``fps`` and ``num_frames`` are None, no frame resampling is
    performed and all frames are returned as-is for the model's processor
    to handle.
    """

    def __init__(
        self,
        *,
        fps: float | None = None,
        max_frames: int | None = None,
        num_frames: int | None = None,
        min_frames: int = DEFAULT_MIN_FRAMES,
        frame_factor: int = DEFAULT_FRAME_FACTOR,
        with_metadata: bool = False,
    ) -> None:
        """Initialize the collator.

        Args:
            fps: Target frames per second for downsampling. The number of
                frames scales with video duration. Only downsamples; if the
                source video has fewer frames than the target, all frames
                are kept.
            max_frames: Safety cap on the number of frames when using
                FPS-based sampling.
            num_frames: If set, use fixed-sample mode: always select this many
                frames uniformly from the video.
            min_frames: Minimum number of frames in FPS-based sampling.
            frame_factor: In FPS-based sampling, round the frame count down to
                a multiple of this.
            with_metadata: Also return a "video_metadata" column with a
                `VideoFrameMetadata` per clip, for processors that need the
                frame timing.
        """
        if num_frames is not None and fps is not None:
            raise ValueError(
                "Cannot specify both `num_frames` and `fps`. "
                "Use `num_frames` for fixed-sample mode or `fps` for FPS-based mode."
            )
        self.fps = fps
        self.max_frames = max_frames
        self.num_frames = num_frames
        self.min_frames = min_frames
        self.frame_factor = frame_factor
        self.with_metadata = with_metadata

    def __call__(self, inputs: list[dict[str, Any]]) -> BatchedInput:
        """Collate a batch of video inputs by resampling the video frames according to the specified parameters."""
        collated_inputs = []
        for row in inputs:
            # assign in place (no pop) to keep the dataset's column order
            row["video"], metadata = self.sample_video(
                row["video"],
                fps=self.fps,
                max_frames=self.max_frames,
                num_frames=self.num_frames,
                min_frames=self.min_frames,
                frame_factor=self.frame_factor,
            )
            if self.with_metadata:
                row["video_metadata"] = metadata
            collated_inputs.append(row)
        return cast(
            "BatchedInput",
            {key: [row[key] for row in collated_inputs] for key in collated_inputs[0]},
        )

    @staticmethod
    def resample_video(
        video: VideoDecoder,
        *,
        fps: float | None = None,
        max_frames: int | None = None,
        num_frames: int | None = None,
        min_frames: int = DEFAULT_MIN_FRAMES,
        frame_factor: int = DEFAULT_FRAME_FACTOR,
    ) -> torch.Tensor:
        """Resample a video input to a target number of frames.

        Same as `sample_video`, without the frame metadata.
        """
        frames, _ = FramesCollator.sample_video(
            video,
            fps=fps,
            max_frames=max_frames,
            num_frames=num_frames,
            min_frames=min_frames,
            frame_factor=frame_factor,
        )
        return frames

    @staticmethod
    def sample_video(
        video: VideoDecoder,
        *,
        fps: float | None = None,
        max_frames: int | None = None,
        num_frames: int | None = None,
        min_frames: int = DEFAULT_MIN_FRAMES,
        frame_factor: int = DEFAULT_FRAME_FACTOR,
    ) -> tuple[torch.Tensor, VideoFrameMetadata]:
        """Resample a video input to a target number of frames and report where they come from.

        When both ``fps`` and ``num_frames`` are None, all frames are returned
        for the model's processor to handle.

        Args:
            video: A VideoDecoder object containing the video data.
            fps: Target frames per second for downsampling. Only
                downsamples; source videos below this rate keep all frames.
            max_frames: Safety cap when using FPS-based sampling.
            num_frames: If set, select exactly this many frames uniformly
                (fixed-sample mode).
            min_frames: Minimum number of frames in FPS-based sampling.
            frame_factor: In FPS-based sampling, round the frame count down to
                a multiple of this.
        """

        def _indices(n_source: int) -> list[int]:
            if num_frames is None and fps is None:
                return list(range(n_source))
            if num_frames is not None:
                target = num_frames
            elif fps is not None:
                duration = video.metadata.end_stream_seconds
                if duration is None:
                    raise RuntimeError(
                        "video has no duration metadata; cannot use FPS-based sampling"
                    )
                target = max(min_frames, int(duration * fps))
                if max_frames is not None:
                    target = min(target, max_frames)
                # only downsample, then round down to a multiple of frame_factor, as
                # qwen-vl-utils does (a 3-frame clip gives 2 frames); wrappers whose
                # processor needs no even count set frame_factor=1 and keep them all
                target = min(target, n_source)
                target = max(frame_factor, target - target % frame_factor)
            else:
                raise RuntimeError("`fps` is not set; cannot use FPS-based sampling")
            return frame_indices(n_source, target)

        # Retry on the actual call: decrement source count when trailing
        # frames fail to decode.
        n = video.metadata.num_frames
        if n is None:
            raise RuntimeError("video has no frame count metadata")
        original_n = n
        while n > 0:
            try:
                indices = _indices(n)
                frames: torch.Tensor = video.get_frames_at(indices).data
                if n < original_n:
                    logger.warning(
                        "Dropped %d undecodable trailing frame(s) from video",
                        original_n - n,
                    )
                metadata = VideoFrameMetadata(
                    total_num_frames=n,
                    fps=getattr(video.metadata, "average_fps", None),
                    duration=getattr(video.metadata, "end_stream_seconds", None),
                    frames_indices=indices,
                )
                return frames, metadata
            except RuntimeError as e:
                if "no more frames" not in str(e):
                    raise
                n -= 1
        raise RuntimeError("video has no decodable frames")


class VideoCollator:
    """Collator that handles any combination of video and audio modalities.

    Uses FramesCollator and AudioCollator static methods to process each modality.
    Supports both FPS-based and fixed-sample video frame selection.
    """

    def __init__(
        self,
        *,
        target_sampling_rate: int,
        fps: float | None = None,
        max_frames: int | None = None,
        num_frames: int | None = None,
        max_samples: int | None = None,
        min_frames: int = DEFAULT_MIN_FRAMES,
        frame_factor: int = DEFAULT_FRAME_FACTOR,
        with_metadata: bool = False,
        resampling: AudioResampling = "kaiser_best",
    ) -> None:
        """Initialize the collator.

        Args:
            target_sampling_rate: The sampling rate to resample audio to.
            fps: Target frames per second for video downsampling.
            max_frames: Safety cap on frames per video for FPS mode.
            num_frames: If set, use fixed-sample mode instead of FPS-based.
            max_samples: Maximum number of audio samples to keep. If None, no truncation.
            min_frames: Minimum number of frames in FPS-based sampling.
            frame_factor: In FPS-based sampling, round the frame count down to
                a multiple of this.
            with_metadata: Also return a "video_metadata" column with a
                `VideoFrameMetadata` per clip (None for rows without video).
            resampling: Audio resampling filter, see `_RESAMPLING_KWARGS`.
        """
        self.fps = fps
        self.max_frames = max_frames
        self.num_frames = num_frames
        self.min_frames = min_frames
        self.frame_factor = frame_factor
        self.with_metadata = with_metadata
        self.target_sampling_rate = target_sampling_rate
        self.max_samples = max_samples
        self.resampling = resampling

    def __call__(self, inputs: list[dict[str, Any]]) -> BatchedInput:
        """Collate a batch of inputs containing video and/or audio by resampling videos and audios according to the specified parameters."""
        collated_inputs = []
        for row in inputs:
            metadata = None
            if "video" in row:
                # assign in place (no pop) to keep the dataset's column order: chat
                # templates that follow the dict order expect video before audio
                row["video"], metadata = FramesCollator.sample_video(
                    row["video"],
                    fps=self.fps,
                    max_frames=self.max_frames,
                    num_frames=self.num_frames,
                    min_frames=self.min_frames,
                    frame_factor=self.frame_factor,
                )
            if self.with_metadata:
                row["video_metadata"] = metadata
            if "audio" in row:
                audio_array = AudioCollator.resample_audio(
                    row,
                    target_sampling_rate=self.target_sampling_rate,
                    max_samples=self.max_samples,
                    resampling=self.resampling,
                )
                row["audio"] = AudioInputItem(
                    array=audio_array, sampling_rate=self.target_sampling_rate
                )
                if "video" in row:
                    # the authors' omni prompts put a clip's video before its audio,
                    # whatever the dataset's column order
                    row["audio"] = row.pop("audio")
            collated_inputs.append(row)
        return cast(
            "BatchedInput",
            {key: [row[key] for row in collated_inputs] for key in collated_inputs[0]},
        )
