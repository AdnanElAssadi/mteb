"""Per-wrapper frame selection, checked through each wrapper's own encode()."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
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
            # with video timing metadata the wrapper passes {"array", "video_metadata"}
            seen.extend(
                row["video"]["array"]
                if isinstance(row["video"], dict)
                else row["video"]
                for row in rows
            )
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


def test_omnivinci_short_clips_use_rounded_linspace() -> None:
    from mteb.models.model_implementations.omnivinci_models import OmniVinciWrapper

    seen: list[Any] = []

    def _encode_batch(batch: dict[str, Any]) -> torch.Tensor:

        seen.extend(batch["video"])
        return torch.zeros(len(batch["video"]), 2)

    wrapper = object.__new__(OmniVinciWrapper)
    wrapper.num_frames = 32
    wrapper.max_audio_samples = None
    wrapper._encode_batch = _encode_batch
    # VILA samples round(linspace(0, n - 1, 32)), also for clips under 32 frames
    wrapper.encode(
        _loader(_StubDecoder(10, seconds=1.0)),
        task_metadata=None,
        hf_split="test",
        hf_subset="default",
    )
    assert seen[0] == [int(i) for i in np.linspace(0, 9, 32).round()]


def test_pe_av_short_clips_keep_every_frame() -> None:
    from mteb.models.model_implementations.pe_av_models import PEAudioVisualWrapper

    seen: list[Any] = []
    wrapper = object.__new__(PEAudioVisualWrapper)
    wrapper.sampling_rate = 48_000
    wrapper.fps, wrapper.max_frames, wrapper.num_frames = 2.0, 64, None
    wrapper.max_samples = None

    def _video(inputs: DataLoader, **kwargs: Any) -> np.ndarray:
        for batch in inputs:
            seen.extend(batch["video"])
        return np.zeros((len(inputs.dataset), 2))

    wrapper.get_video_embeddings = _video
    # PE-AV takes any frame count: a 3-frame clip keeps 3 frames, a 23.6 s clip 47
    wrapper.encode(
        _loader(_StubDecoder(3, seconds=0.1), _StubDecoder(708, seconds=23.6)),
        task_metadata=None,
        hf_split="test",
        hf_subset="default",
    )
    assert seen[0] == [0, 1, 2]
    assert len(seen[1]) == 47
