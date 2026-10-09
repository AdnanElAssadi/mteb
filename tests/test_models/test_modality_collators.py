from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from torch.utils.data import DataLoader

from mteb._create_dataloaders import _custom_collate_fn
from mteb.models.modality_collators import (
    AudioCollator,
    FramesCollator,
    VideoCollator,
    frame_indices,
    seconds_to_samples,
    single_clip_dataloader,
)


class _StubDecoder:
    """Minimal VideoDecoder stand-in: returns the requested indices as frames."""

    def __init__(self, num_frames: int, seconds: float) -> None:
        self.metadata = SimpleNamespace(
            num_frames=num_frames,
            end_stream_seconds=seconds,
            average_fps=num_frames / seconds,
        )

    def get_frames_at(self, indices: list[int]) -> SimpleNamespace:
        return SimpleNamespace(data=list(indices))


@pytest.mark.parametrize(
    ("n_source", "num_frames", "expected"),
    [
        # evenly spaced over the whole clip, not just its head
        (15, 8, [0, 2, 4, 6, 8, 10, 12, 14]),
        (300, 8, [0, 43, 85, 128, 171, 214, 256, 299]),
        # fewer source frames than requested: repeat each frame in place
        (5, 8, [0, 0, 1, 1, 2, 3, 3, 4]),
        (3, 8, [0, 0, 0, 1, 1, 1, 2, 2]),
        (1, 4, [0, 0, 0, 0]),
        (16, 16, list(range(16))),
    ],
)
def test_resample_video_fixed_frames(
    n_source: int, num_frames: int, expected: list[int]
) -> None:
    video = _StubDecoder(n_source, seconds=10.0)
    assert FramesCollator.resample_video(video, num_frames=num_frames) == expected


def test_resample_video_fps_spans_clip_and_respects_cap() -> None:
    video = _StubDecoder(300, seconds=10.0)
    frames = FramesCollator.resample_video(video, fps=2.0)
    assert len(frames) == 20
    assert frames[0] == 0
    assert frames[-1] == 299
    capped = FramesCollator.resample_video(video, fps=2.0, max_frames=8)
    assert capped == [0, 43, 85, 128, 171, 214, 256, 299]


@pytest.mark.parametrize(
    ("n_source", "seconds", "fps", "max_frames", "expected"),
    [
        # under 2 s at 1 fps: 4 frames, not 1 (a Qwen3-VL processor needs >= 2)
        (11, 0.37, 1.0, 64, [0, 3, 7, 10]),
        (30, 1.2, 2.0, 64, [0, 10, 19, 29]),
        # odd counts are rounded down to even
        (146, 6.09, 0.5, 64, [0, 48, 97, 145]),
        (75, 2.5, 2.0, 64, [0, 25, 49, 74]),
        # fewer source frames than the minimum: rounded down to an even count,
        # like qwen-vl-utils
        (3, 0.1, 2.0, 64, [0, 2]),
        (1, 0.04, 2.0, 64, [0, 0]),
    ],
)
def test_resample_video_fps_min_and_even_frames(
    n_source: int,
    seconds: float,
    fps: float,
    max_frames: int,
    expected: list[int],
) -> None:
    video = _StubDecoder(n_source, seconds=seconds)
    frames = FramesCollator.resample_video(video, fps=fps, max_frames=max_frames)
    assert frames == expected


def test_resample_video_fps_odd_count_rounded_down() -> None:
    video = _StubDecoder(1350, seconds=45.0)
    assert len(FramesCollator.resample_video(video, fps=1.0, max_frames=64)) == 44
    assert len(FramesCollator.resample_video(video, fps=1.0, max_frames=33)) == 32


def test_resample_video_fps_without_min_frames() -> None:
    """Re-encoding callers (API models) opt out to keep the clip duration."""
    video = _StubDecoder(11, seconds=0.37)
    frames = FramesCollator.resample_video(video, fps=1.0, min_frames=1, frame_factor=1)
    assert frames == [0]


def test_resample_video_without_sampling_keeps_all_frames() -> None:
    video = _StubDecoder(12, seconds=1.0)
    assert FramesCollator.resample_video(video) == list(range(12))


def test_seconds_to_samples() -> None:
    assert seconds_to_samples(None, 16_000) is None
    assert seconds_to_samples(1.5, 16_000) == 24_000
    with pytest.raises(ValueError, match="must be positive"):
        seconds_to_samples(0, 16_000)


def test_single_clip_dataloader() -> None:
    rows = [{"audio": {"array": [float(i)], "sampling_rate": 16_000}} for i in range(5)]
    loader = DataLoader(rows, batch_size=2, collate_fn=_custom_collate_fn)  # type: ignore[arg-type]
    single = single_clip_dataloader(loader, "test")
    assert single.batch_size == 1
    assert len(single) == 5
    batches = list(single)
    assert [len(b["audio"]) for b in batches] == [1] * 5
    assert [b["audio"][0]["array"] for b in batches] == [[float(i)] for i in range(5)]


def test_audio_collator_rejects_non_positive_cap() -> None:
    audio = {"array": [0.0] * 10, "sampling_rate": 16_000}
    with pytest.raises(ValueError, match="must be positive"):
        AudioCollator.resample_audio(audio, target_sampling_rate=16_000, max_samples=0)


@pytest.mark.parametrize(
    ("mode", "n_source", "expected"),
    [
        # 8 frames from a 25-frame and a 5-frame clip
        ("default", 25, [0, 3, 7, 10, 14, 17, 21, 24]),
        ("default", 5, [0, 0, 1, 1, 2, 3, 3, 4]),
        ("linspace_round", 25, [0, 3, 7, 10, 14, 17, 21, 24]),
        ("linspace_round", 5, [0, 1, 1, 2, 2, 3, 3, 4]),
        # Cosmos-Embed1 card / LanguageBind: np.linspace(0, n - 1, 8, dtype=int)
        ("linspace_floor", 25, [0, 3, 6, 10, 13, 17, 20, 24]),
        ("linspace_floor", 5, [0, 0, 1, 1, 2, 2, 3, 4]),
        # Jina v5 omni: clips with at most 8 (there: 32) frames keep all of them
        ("linspace_floor_no_pad", 25, [0, 3, 6, 10, 13, 17, 20, 24]),
        ("linspace_floor_no_pad", 5, [0, 1, 2, 3, 4]),
        # VideoPrism: np.linspace(0, n, 8, endpoint=False, dtype=np.int32)
        ("uniform", 25, [0, 3, 6, 9, 12, 15, 18, 21]),
        ("uniform", 5, [0, 0, 1, 1, 2, 3, 3, 4]),
        # X-CLIP: mmaction SampleFrames(test_mode=True), segment middles
        ("segment_middle", 25, [1, 4, 7, 10, 14, 17, 20, 23]),
        ("segment_middle", 5, [0, 0, 1, 2, 2, 3, 4, 4]),
        # InternVideo2: get_frame_indices(sample="middle"), pad with the last frame
        ("segment_middle_pad_last", 25, [1, 4, 7, 10, 13, 16, 19, 22]),
        ("segment_middle_pad_last", 5, [0, 1, 2, 3, 4, 4, 4, 4]),
        # V-JEPA2 / VideoMAE: short clips repeat their last frame
        ("pad_last", 25, [0, 3, 7, 10, 14, 17, 21, 24]),
        ("pad_last", 5, [0, 1, 2, 3, 4, 4, 4, 4]),
    ],
)
def test_frame_index_modes(mode: str, n_source: int, expected: list[int]) -> None:
    assert frame_indices(n_source, 8, mode) == expected  # type: ignore[arg-type]
    video = _StubDecoder(n_source, seconds=1.0)
    assert (
        FramesCollator.resample_video(video, num_frames=8, frame_index_mode=mode)  # type: ignore[arg-type]
        == expected
    )


def test_frame_index_mode_reaches_collators() -> None:
    rows = [{"video": _StubDecoder(5, seconds=1.0)}]
    out = FramesCollator(num_frames=8, frame_index_mode="linspace_round")(rows)
    assert out["video"] == [[0, 1, 1, 2, 2, 3, 3, 4]]
    rows = [{"video": _StubDecoder(5, seconds=1.0)}]
    out = VideoCollator(
        target_sampling_rate=16000, num_frames=8, frame_index_mode="linspace_round"
    )(rows)
    assert out["video"] == [[0, 1, 1, 2, 2, 3, 3, 4]]
    with pytest.raises(ValueError, match="Unknown frame index mode"):
        frame_indices(5, 8, "nope")  # type: ignore[arg-type]


def test_frame_index_mode_callable_gets_source_fps() -> None:
    calls = []

    def _sampler(n_source: int, target: int, source_fps: float | None) -> list[int]:
        calls.append((n_source, target, source_fps))
        return [0] * target

    video = _StubDecoder(50, seconds=2.0)
    assert FramesCollator.resample_video(
        video, num_frames=3, frame_index_mode=_sampler
    ) == [0, 0, 0]
    assert calls == [(50, 3, 25.0)]


def test_audio_downmix() -> None:
    stereo = np.stack([np.ones(160), -np.ones(160)]).astype(np.float32)
    row = {"audio": {"array": stereo, "sampling_rate": 16_000}}
    mean = AudioCollator.resample_audio(row, target_sampling_rate=16_000)
    first = AudioCollator.resample_audio(
        row, target_sampling_rate=16_000, downmix="first"
    )
    np.testing.assert_array_equal(mean, np.zeros(160))
    np.testing.assert_array_equal(first, np.ones(160))
    out = AudioCollator(target_sampling_rate=16_000, downmix="first")([dict(row)])
    np.testing.assert_array_equal(out["audio"][0]["array"], np.ones(160))
    out = VideoCollator(target_sampling_rate=16_000, downmix="first")([dict(row)])
    np.testing.assert_array_equal(out["audio"][0]["array"], np.ones(160))
    out = AudioCollator(target_sampling_rate=16_000)([dict(row)])
    np.testing.assert_array_equal(out["audio"][0]["array"], np.zeros(160))


class _StubAudioDecoder:
    """datasets' AudioDecoder: ``["array"]`` averages the channels."""

    def __init__(self, samples: np.ndarray, sampling_rate: int) -> None:
        self.samples, self.sampling_rate = samples, sampling_rate

    def __getitem__(self, key: str) -> object:
        if key == "array":
            return self.samples.mean(axis=0)
        return self.sampling_rate

    def get_all_samples(self) -> SimpleNamespace:
        import torch

        return SimpleNamespace(data=torch.from_numpy(self.samples))


@pytest.mark.parametrize("sampling_rate", [16_000, 32_000])
def test_audio_downmix_first_reads_decoder_channels(sampling_rate: int) -> None:
    stereo = np.stack([np.ones(320), -np.ones(320)]).astype(np.float32)
    row = {"audio": _StubAudioDecoder(stereo, sampling_rate)}
    first = AudioCollator.resample_audio(
        row, target_sampling_rate=16_000, downmix="first"
    )
    mean = AudioCollator.resample_audio(row, target_sampling_rate=16_000)
    assert first.ndim == 1
    assert first.mean() > 0.9
    np.testing.assert_allclose(mean, 0.0, atol=1e-6)
    # mono files decode to one channel and stay 1-D
    row = {"audio": _StubAudioDecoder(stereo[:1], sampling_rate)}
    mono = AudioCollator.resample_audio(
        row, target_sampling_rate=16_000, downmix="first"
    )
    assert mono.ndim == 1


def test_resample_video_fps_short_source_frame_counts() -> None:
    """Short sources: frame_factor=1 keeps every frame, frame_factor=2 rounds down
    to an even count like qwen-vl-utils (but never below 2 frames)."""
    for n_source in range(1, 8):
        video = _StubDecoder(n_source, seconds=0.2)
        frames = FramesCollator.resample_video(
            video, fps=2.0, max_frames=64, frame_factor=1
        )
        assert len(frames) >= min(n_source, 4)
        assert set(frames) == set(range(n_source)) or n_source > 4
        frames = FramesCollator.resample_video(
            video, fps=2.0, max_frames=64, frame_factor=2
        )
        assert len(frames) == max(2, min(n_source, 4) // 2 * 2)
    video = _StubDecoder(3, seconds=0.1)
    assert FramesCollator.resample_video(video, fps=2.0, frame_factor=1) == [0, 1, 2]
    assert FramesCollator.resample_video(video, fps=2.0, frame_factor=2) == [0, 2]
