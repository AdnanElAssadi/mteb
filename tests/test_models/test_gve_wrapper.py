from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
import transformers
from transformers import Qwen2VLVideoProcessor

from mteb.models.model_implementations.gve_models import GVEWrapper, gve_3b


class _FakeModel:
    def eval(self) -> _FakeModel:
        return self

    def to(self, *args: Any, **kwargs: Any) -> _FakeModel:
        return self


@pytest.fixture
def wrapper(monkeypatch: pytest.MonkeyPatch) -> GVEWrapper:
    monkeypatch.setattr(
        transformers.Qwen2_5_VLForConditionalGeneration,
        "from_pretrained",
        lambda *a, **k: _FakeModel(),
    )
    processor = SimpleNamespace(tokenizer=SimpleNamespace(padding_side="right"))
    monkeypatch.setattr(
        transformers.AutoProcessor, "from_pretrained", lambda *a, **k: processor
    )
    return GVEWrapper(gve_3b.name, revision=gve_3b.revision, device="cpu")


@pytest.mark.parametrize(
    ("height", "width", "expected_grid"),
    [
        # low-res frames are upscaled to qwen-vl-utils' 128-token floor, as in
        # the model card's video path
        (240, 320, [4, 20, 28]),
        # large frames are capped at max_pixels=200*28*28 per frame
        (1080, 1920, [4, 20, 36]),
    ],
)
def test_gve_video_pixel_budget(
    wrapper: GVEWrapper, height: int, width: int, expected_grid: list[int]
) -> None:
    video = torch.randint(0, 256, (8, 3, height, width), dtype=torch.uint8)
    out = Qwen2VLVideoProcessor()(
        videos=[video], return_tensors="pt", **wrapper.videos_kwargs
    )
    assert out["video_grid_thw"].tolist() == [expected_grid]
