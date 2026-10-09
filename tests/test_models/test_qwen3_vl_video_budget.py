from __future__ import annotations

import pytest
import torch
from transformers import Qwen3VLVideoProcessor

import mteb
from mteb.models.model_implementations.qwen3_vl_embedding_models import (
    VIDEO_MAX_TOTAL_PIXELS,
    load_frame_budget_video_processor,
)


@pytest.fixture
def video_processor(monkeypatch: pytest.MonkeyPatch) -> Qwen3VLVideoProcessor:
    # default Qwen3-VL video processor config (patch 16, merge 2), no download
    monkeypatch.setattr(
        Qwen3VLVideoProcessor,
        "from_pretrained",
        classmethod(lambda cls, *args, **kwargs: cls(do_sample_frames=False)),
    )
    return load_frame_budget_video_processor(
        "Qwen/Qwen3-VL-Embedding-2B", total_pixels=VIDEO_MAX_TOTAL_PIXELS
    )


@pytest.mark.parametrize(
    ("frames", "height", "width", "expected_grid"),
    [
        # grids of the official script (qwen-vl-utils) on the same clips
        (2, 1080, 1920, [1, 40, 72]),
        (10, 240, 320, [5, 20, 28]),
        (6, 360, 640, [3, 22, 40]),
        # 64 frames: the clip budget (total_pixels) lowers the per-frame size
        (64, 1080, 1920, [32, 22, 40]),
        # a single frame is resized per frame too (no whole-clip smart_resize)
        (1, 240, 320, [1, 20, 28]),
    ],
)
def test_qwen3_vl_video_frame_budget(
    video_processor: Qwen3VLVideoProcessor,
    frames: int,
    height: int,
    width: int,
    expected_grid: list[int],
) -> None:
    video = torch.randint(0, 256, (frames, 3, height, width), dtype=torch.uint8)
    out = video_processor(videos=[video], return_tensors="pt")
    assert out["video_grid_thw"].tolist() == [expected_grid]


@pytest.mark.parametrize(
    "name", ["Qwen/Qwen3-VL-Reranker-2B", "Qwen/Qwen3-VL-Reranker-8B"]
)
def test_qwen3_vl_rerankers_use_frame_budget(name: str) -> None:
    # both rerankers share the official script's qwen-vl-utils per-frame budget
    meta = mteb.get_model_meta(name)
    assert (meta.loader_kwargs or {}).get("video_frame_budget") is True
