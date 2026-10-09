from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import transformers

from mteb.models.model_implementations.vlm2vec_models import VLM2VEC2Wrapper, vlm2vec2

# preprocessor_config.json of VLM2Vec/VLM2Vec-V2.0 at the pinned revision
CHECKPOINT_MAX_PIXELS = 1003520


class _FakeModel:
    def merge_and_unload(self) -> _FakeModel:
        return self

    def to(self, *args: Any, **kwargs: Any) -> _FakeModel:
        return self

    def eval(self) -> _FakeModel:
        return self


def test_vlm2vec_v2_keeps_checkpoint_image_pixels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor = SimpleNamespace(
        image_processor=SimpleNamespace(max_pixels=CHECKPOINT_MAX_PIXELS)
    )
    monkeypatch.setattr(
        transformers.AutoProcessor, "from_pretrained", lambda *a, **k: processor
    )
    monkeypatch.setattr(
        transformers.Qwen2VLForConditionalGeneration,
        "from_pretrained",
        lambda *a, **k: _FakeModel(),
    )
    peft = ModuleType("peft")
    peft.PeftModel = SimpleNamespace(from_pretrained=lambda *a, **k: _FakeModel())
    monkeypatch.setitem(sys.modules, "peft", peft)

    wrapper = VLM2VEC2Wrapper(vlm2vec2.name, revision=vlm2vec2.revision, device="cpu")

    # the official eval resizes images with the checkpoint's limit; a 360*420
    # cap shrinks a 1080p image from grid [1,52,94] to [1,20,36]
    assert wrapper.processor.image_processor.max_pixels == CHECKPOINT_MAX_PIXELS
