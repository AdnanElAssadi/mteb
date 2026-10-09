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
    seconds_to_samples,
    single_clip_dataloader,
)


class _StubDecoder:
    """Minimal VideoDecoder stand-in: returns the requested indices as frames."""

    def __init__(self, num_frames: int, seconds: float) -> None:
        self.metadata = SimpleNamespace(
            num_frames=num_frames, end_stream_seconds=seconds
        )

    def get_frames_at(self, indices: list[int]) -> SimpleNamespace:
        return SimpleNamespace(data=list(indices))


@pytest.mark.parametrize(
    ("n_source", "num_frames", "expected"),
    [
        # evenly spaced over the whole clip, not just its head
        (15, 8, [0, 2, 4, 6, 8, 10, 12, 14]),
        (300, 8, [0, 43, 85, 128, 171, 214, 256, 299]),
        # fewer source frames than requested: repeat to reach the count
        (5, 8, [0, 1, 2, 3, 4, 0, 1, 2]),
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
