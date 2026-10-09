from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
import transformers
from transformers import Qwen2VLVideoProcessor

from mteb.models.model_implementations import unite_models
from mteb.models.model_implementations.unite_models import (
    UniteWrapper,
    unite_base_qwen2vl_2b,
)


class _FakeModel:
    def eval(self) -> _FakeModel:
        return self

    def to(self, *args: Any, **kwargs: Any) -> _FakeModel:
        return self


@pytest.mark.parametrize(
    ("height", "width", "expected_grid"),
    [
        # small frames are upscaled to qwen_vl_utils' VIDEO_MIN_PIXELS=128*28*28,
        # as in UNITE's reference inference
        (240, 320, [16, 20, 28]),
        # large frames are capped at 360*420
        (1080, 1920, [16, 20, 36]),
    ],
)
def test_unite_video_pixel_budget(
    height: int, width: int, expected_grid: list[int], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        unite_models, "_build_unite_model", lambda *a, **k: _FakeModel()
    )
    # the checkpoint's video processor keeps the Qwen2-VL image limits
    video_processor = Qwen2VLVideoProcessor(
        size={"shortest_edge": 56 * 56, "longest_edge": 28 * 28 * 16384}
    )
    processor = SimpleNamespace(video_processor=video_processor)
    monkeypatch.setattr(
        transformers.AutoProcessor, "from_pretrained", lambda *a, **k: processor
    )
    wrapper = UniteWrapper(
        unite_base_qwen2vl_2b.name, unite_base_qwen2vl_2b.revision, device="cpu"
    )

    video = torch.randint(0, 256, (32, 3, height, width), dtype=torch.uint8)
    out = wrapper.processor.video_processor(
        videos=[video], do_sample_frames=False, return_tensors="pt"
    )
    assert out["video_grid_thw"].tolist() == [expected_grid]
