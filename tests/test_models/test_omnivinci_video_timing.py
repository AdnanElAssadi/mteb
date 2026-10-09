from __future__ import annotations

from fractions import Fraction

import pytest
import torch

from mteb.models.model_implementations.omnivinci_models import OmniVinciWrapper


@pytest.fixture
def written_fps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    torchvision = pytest.importorskip("torchvision")
    calls: list[float] = []
    monkeypatch.setattr(
        torchvision.io,
        "write_video",
        lambda path, frames, fps: calls.append(fps),
        raising=False,
    )
    return calls


def test_temp_video_uses_the_real_frame_spacing(written_fps: list[float]) -> None:
    frames = torch.zeros(32, 3, 4, 4, dtype=torch.uint8)
    # 32 frames spread over a 45 s, 30 fps clip
    indices = [round(i * 1349 / 31) for i in range(32)]
    metadata = {
        "total_num_frames": 1350,
        "fps": 30.0,
        "duration": 45.0,
        "frames_indices": indices,
    }
    OmniVinciWrapper._save_frames_as_video(frames, metadata)  # type: ignore[arg-type]
    (fps,) = written_fps
    assert isinstance(fps, Fraction)  # write_video rounds floats to integers
    assert 31 / float(fps) == pytest.approx(1349 / 30, rel=1e-3)


def test_temp_video_without_metadata_keeps_24_fps(written_fps: list[float]) -> None:
    OmniVinciWrapper._save_frames_as_video(torch.zeros(2, 3, 4, 4, dtype=torch.uint8))
    assert written_fps == [24]
