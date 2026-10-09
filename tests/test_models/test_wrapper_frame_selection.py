"""Per-wrapper frame selection, checked through each wrapper's own encode()."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
from torch.utils.data import DataLoader, Dataset


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


class _VideoDataset(Dataset):
    features = {"video": None}

    def __init__(self, videos: list[_StubDecoder]) -> None:
        self.videos = videos

    def __len__(self) -> int:
        return len(self.videos)

    def __getitem__(self, i: int) -> dict[str, Any]:
        return {"video": self.videos[i]}


def _loader(*videos: _StubDecoder) -> DataLoader:
    return DataLoader(_VideoDataset(list(videos)), batch_size=len(videos))


def test_bidirlm_keeps_odd_fps_frame_counts() -> None:
    from mteb.models.model_implementations.bidirlm_omni_models import (
        BidirLMOmniEncoder,
    )

    seen: list[Any] = []

    class _Model:
        def encode(self, rows: list[dict[str, Any]], **kwargs: Any) -> np.ndarray:
            seen.extend(row["video"] for row in rows)
            return np.zeros((len(rows), 2))

    wrapper = object.__new__(BidirLMOmniEncoder)
    wrapper.model = _Model()
    wrapper.sampling_rate = 16_000
    wrapper.fps, wrapper.max_frames, wrapper.num_frames = 2.0, 64, None
    wrapper.max_samples = None
    wrapper.max_text_length = 1024
    wrapper._get_instruction = lambda *args: ""
    # 23.6 s at 2 fps = 47 frames; the Qwen3-VL processor pads the 47th itself
    wrapper.encode(
        _loader(_StubDecoder(708, seconds=23.6)),
        task_metadata=None,
        hf_split="test",
        hf_subset="default",
    )
    assert len(seen[0]) == 47
