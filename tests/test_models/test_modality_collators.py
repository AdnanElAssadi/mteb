from __future__ import annotations

from types import SimpleNamespace

import pytest
from torch.utils.data import DataLoader

from mteb._create_dataloaders import _custom_collate_fn
from mteb.models.modality_collators import (
    AudioCollator,
    FramesCollator,
    VideoCollator,
    fps_shortfall,
    sampled_fps,
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
        # fewer source frames than the minimum: keep an even count
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


def test_collators_return_frame_metadata() -> None:
    rows = [{"video": _StubDecoder(1350, seconds=45.0)}]
    batch = FramesCollator(fps=2.0, max_frames=8, with_metadata=True)(rows)
    assert batch["video"] == [[0, 193, 385, 578, 771, 964, 1156, 1349]]
    assert batch["video_metadata"] == [
        {
            "total_num_frames": 1350,
            "fps": 30.0,
            "duration": 45.0,
            "frames_indices": [0, 193, 385, 578, 771, 964, 1156, 1349],
        }
    ]

    row = {"video": _StubDecoder(30, seconds=1.2), "text": "a"}
    batch = VideoCollator(target_sampling_rate=16_000, fps=1.0, with_metadata=True)(
        [dict(row)]
    )
    assert batch["video_metadata"][0]["frames_indices"] == [0, 10, 19, 29]
    plain = VideoCollator(target_sampling_rate=16_000, fps=1.0)([dict(row)])
    assert "video_metadata" not in plain


def test_metadata_feeds_transformers_video_metadata() -> None:
    video_utils = pytest.importorskip("transformers.video_utils")
    _, metadata = FramesCollator.sample_video(_StubDecoder(300, seconds=10.0), fps=1.0)
    converted = video_utils.VideoMetadata(**metadata)
    # real times of the sampled frames, not frame_index / 24 fps
    assert metadata["frames_indices"] == [0, 33, 66, 100, 133, 166, 199, 233, 266, 299]
    assert converted.timestamps == pytest.approx(
        [i / 30 for i in metadata["frames_indices"]]
    )


@pytest.mark.parametrize(
    ("n_source", "seconds", "fps", "max_frames", "shortfall"),
    [
        # normal clip: 12 frames for 6.09 s at 2 fps, only rounding
        (146, 6.09, 2.0, 64, 1.0),
        # short clip padded to 4 frames
        (30, 1.2, 2.0, 64, 1.0),
        # 45 s capped at 64 frames
        (1350, 45.0, 2.0, 64, 2.0 / (64 / 45.0)),
        # 1 fps source asked for 2 fps
        (10, 10.0, 2.0, 64, 2.0),
    ],
)
def test_fps_shortfall(
    n_source: int, seconds: float, fps: float, max_frames: int, shortfall: float
) -> None:
    _, metadata = FramesCollator.sample_video(
        _StubDecoder(n_source, seconds=seconds), fps=fps, max_frames=max_frames
    )
    assert sampled_fps(metadata) == pytest.approx(
        len(metadata["frames_indices"]) / seconds
    )
    assert fps_shortfall(metadata, fps) == pytest.approx(shortfall)


def test_resample_video_without_sampling_keeps_all_frames() -> None:
    video = _StubDecoder(12, seconds=1.0)
    assert FramesCollator.resample_video(video) == list(range(12))


@pytest.mark.parametrize(
    "collator",
    [
        VideoCollator(target_sampling_rate=16_000, num_frames=2),
        FramesCollator(num_frames=2),
    ],
)
def test_video_collators_keep_column_order(
    collator: VideoCollator | FramesCollator,
) -> None:
    """Chat templates that follow the dict order must see video before audio."""
    row = {
        "id": "a",
        "video": _StubDecoder(4, seconds=1.0),
        "text": "caption",
    }
    if isinstance(collator, VideoCollator):
        row["audio"] = {"array": [0.0] * 10, "sampling_rate": 16_000}
    columns = list(row)
    batch = collator([row])
    assert list(batch) == columns
    assert batch["video"] == [[0, 3]]


def test_video_collator_puts_video_before_audio() -> None:
    """Even when the dataset lists audio first, video must come before audio."""
    collator = VideoCollator(target_sampling_rate=16_000, num_frames=2)
    row = {
        "id": "a",
        "audio": {"array": [0.0] * 10, "sampling_rate": 16_000},
        "video": _StubDecoder(4, seconds=1.0),
        "text": "caption",
    }
    batch = collator([row])
    assert list(batch) == ["id", "video", "text", "audio"]


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


def test_sentence_transformer_dicts_carry_video_metadata() -> None:
    from mteb.models.sentence_transformer_wrapper import _batch_to_modality_dicts

    rows = [{"video": _StubDecoder(30, seconds=1.2), "text": "a", "id": "x"}]
    batch = VideoCollator(target_sampling_rate=16_000, fps=1.0, with_metadata=True)(
        rows
    )
    (sample,) = _batch_to_modality_dicts(batch, ["video", "text"])
    assert sample["text"] == "a"
    assert sample["video"]["array"] == [0, 10, 19, 29]
    assert sample["video"]["video_metadata"]["fps"] == 25.0

    rows = [{"video": _StubDecoder(30, seconds=1.2), "text": "a"}]
    batch = VideoCollator(target_sampling_rate=16_000, fps=1.0)(rows)
    (sample,) = _batch_to_modality_dicts(batch, ["video", "text"])
    assert sample["video"] == [0, 10, 19, 29]
