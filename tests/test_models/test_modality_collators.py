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
    fps_shortfall,
    frame_indices,
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


def _resampled_tone_level(frequency: float, resampling: str) -> float:
    """RMS of a unit tone after 24 kHz -> 16 kHz, relative to the input tone."""
    t = np.arange(24_000) / 24_000
    tone = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    out = AudioCollator.resample_audio(
        {"audio": {"array": tone, "sampling_rate": 24_000}},
        target_sampling_rate=16_000,
        resampling=resampling,  # type: ignore[arg-type]
    )
    return float(np.sqrt(np.mean(out[1000:-1000] ** 2) * 2))


def test_audio_collator_resampling_filter() -> None:
    # default: a sharp low-pass like librosa/soxr. A tone above the new Nyquist
    # (8 kHz) is removed instead of aliasing back, one below it is kept.
    assert _resampled_tone_level(8_500, "kaiser_best") < 0.01
    assert _resampled_tone_level(7_000, "kaiser_best") > 0.99
    # torchaudio's default filter, kept for models whose authors use it
    assert _resampled_tone_level(8_500, "torchaudio_default") > 0.2
    assert _resampled_tone_level(7_000, "torchaudio_default") < 0.9


def test_audio_collator_torchaudio_default_matches_torchaudio() -> None:
    import torch
    import torchaudio

    audio = np.random.default_rng(0).standard_normal(4_410).astype(np.float32)
    out = AudioCollator.resample_audio(
        {"audio": {"array": audio, "sampling_rate": 44_100}},
        target_sampling_rate=16_000,
        resampling="torchaudio_default",
    )
    expected = torchaudio.transforms.Resample(44_100, 16_000)(torch.from_numpy(audio))
    np.testing.assert_array_equal(out, expected.numpy())


def test_video_collator_resampling_filter() -> None:
    tone = np.sin(2 * np.pi * 8_500 * np.arange(24_000) / 24_000).astype(np.float32)
    rows = [{"audio": {"array": tone, "sampling_rate": 24_000}}]
    levels = {}
    for resampling in ("kaiser_best", "torchaudio_default"):
        out = VideoCollator(target_sampling_rate=16_000, resampling=resampling)(  # type: ignore[arg-type]
            [dict(row) for row in rows]
        )
        array = out["audio"][0]["array"]
        levels[resampling] = float(np.sqrt(np.mean(array[1000:-1000] ** 2) * 2))
    assert levels["kaiser_best"] < 0.01
    assert levels["torchaudio_default"] > 0.2


def test_sentence_transformer_collator_gets_audio_resampling() -> None:
    from mteb.models.sentence_transformer_wrapper import _setup_modality_collator

    class _Dataset:
        features = {"audio": None}

    for feature in ("audio", "video"):
        _Dataset.features = {feature: None}
        loader = SimpleNamespace(dataset=_Dataset(), collate_fn=None)
        _setup_modality_collator(
            loader,  # type: ignore[arg-type]
            fps=None,
            max_frames=None,
            num_frames=None,
            target_sampling_rate=None,
            max_samples=None,
            resampling="torchaudio_default",
        )
        assert loader.collate_fn.resampling == "torchaudio_default"


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


@pytest.mark.parametrize(
    ("n_source", "expected"),
    [
        # 8 frames from a 25-frame and a 5-frame clip
        (25, [0, 3, 7, 10, 14, 17, 21, 24]),
        (5, [0, 0, 1, 1, 2, 3, 3, 4]),
    ],
)
def test_frame_indices(n_source: int, expected: list[int]) -> None:
    assert frame_indices(n_source, 8) == expected
    video = _StubDecoder(n_source, seconds=1.0)
    assert FramesCollator.resample_video(video, num_frames=8) == expected


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


def test_stretch_second_per_grid_only_for_slow_clips() -> None:
    import torch

    from mteb.models.modality_collators import stretch_second_per_grid

    clips = [
        _StubDecoder(146, seconds=6.09),  # normal clip at 2 fps
        _StubDecoder(1350, seconds=45.0),  # capped at 64 frames
        _StubDecoder(30, seconds=1.2),  # padded to 4 frames
    ]
    metadata = [
        FramesCollator.sample_video(clip, fps=2.0, max_frames=64)[1] for clip in clips
    ]
    stretched = stretch_second_per_grid(
        torch.tensor([1.0, 1.0, 1.0], dtype=torch.bfloat16), metadata, 2.0
    )
    assert stretched.dtype == torch.float32
    assert stretched.tolist() == pytest.approx([1.0, 2.0 / (64 / 45.0), 1.0])


def test_omni_preprocess_hook_stretches_per_clip() -> None:
    import torch

    from mteb.models.sentence_transformer_wrapper import _stretch_omni_video_timing

    class _Module:
        def preprocess(self, inputs, prompt=None):
            n = sum("video" in item for item in inputs)
            return {"video_second_per_grid": torch.ones(n)}

    module = _Module()
    _stretch_omni_video_timing([module], fps=2.0)  # type: ignore[arg-type]
    _, capped = FramesCollator.sample_video(
        _StubDecoder(1350, seconds=45.0), fps=2.0, max_frames=64
    )
    _, normal = FramesCollator.sample_video(_StubDecoder(146, seconds=6.09), fps=2.0)
    features = module.preprocess(
        [
            {"text": "no video"},
            {"video": {"array": [], "video_metadata": capped}},
            {"video": {"array": [], "video_metadata": normal}},
        ]
    )
    assert features["video_second_per_grid"].tolist() == pytest.approx(
        [2.0 / (64 / 45.0), 1.0]
    )
