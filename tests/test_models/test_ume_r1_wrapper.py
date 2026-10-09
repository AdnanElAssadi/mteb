from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
import transformers
from PIL import Image

from mteb.models.model_implementations.ume_r1 import (
    UME_R1_EMBED_PROMPT,
    UMER1Wrapper,
    ume_r1_2b,
)

SUFFIX = f"\n<disc_emb>\n{UME_R1_EMBED_PROMPT}"


class _FakeModel:
    generation_config = SimpleNamespace(eos_token_id=0)

    def to(self, *args: Any, **kwargs: Any) -> _FakeModel:
        return self

    def eval(self) -> _FakeModel:
        return self


@pytest.fixture
def loaded(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    calls: dict[str, Any] = {}

    def model_from_pretrained(*args: Any, **kwargs: Any) -> _FakeModel:
        calls["model"] = kwargs
        return _FakeModel()

    def processor_from_pretrained(*args: Any, **kwargs: Any) -> SimpleNamespace:
        calls["processor"] = kwargs
        return SimpleNamespace(
            tokenizer=SimpleNamespace(padding_side="right", get_vocab=dict),
            video_processor=SimpleNamespace(size={}),
        )

    monkeypatch.setattr(
        transformers.Qwen2VLForConditionalGeneration,
        "from_pretrained",
        model_from_pretrained,
    )
    monkeypatch.setattr(
        transformers.AutoProcessor, "from_pretrained", processor_from_pretrained
    )
    calls["wrapper"] = UMER1Wrapper(
        ume_r1_2b.name, revision=ume_r1_2b.revision, device="cpu"
    )
    return calls


def test_ume_r1_loads_official_pixel_settings(loaded: dict[str, Any]) -> None:
    # model weights are pinned, and images keep the checkpoint's 784-2359296
    # pixel limits as in the official eval
    assert loaded["model"]["revision"] == ume_r1_2b.revision
    assert "min_pixels" not in loaded["processor"]
    assert "max_pixels" not in loaded["processor"]
    # qwen-vl-utils' per-frame video size
    assert loaded["wrapper"].processor.video_processor.size == {
        "shortest_edge": 128 * 28 * 28,
        "longest_edge": 768 * 28 * 28,
    }


def test_ume_r1_prompt_layout_matches_eval_collator() -> None:
    image = Image.new("RGB", (64, 64))
    video = torch.zeros(4, 3, 64, 64)
    text_msg, image_msg, video_msg = (
        UMER1Wrapper._build_messages(batch)[0][0]["content"]
        for batch in (
            {"text": ["a cat"]},
            {"text": ["a cat"], "image": [image]},
            {"video": [video]},
        )
    )
    # text only: blank white image first (as qwen-vl-utils resizes it)
    assert text_msg[0]["type"] == "image"
    assert text_msg[0]["image"].size == (56, 56)
    assert text_msg[1] == {"type": "text", "text": "a cat" + SUFFIX}
    # media first, then the text and prompt in one item
    assert image_msg == [
        {"type": "image", "image": image},
        {"type": "text", "text": "a cat" + SUFFIX},
    ]
    assert video_msg == [
        {"type": "video", "video": video},
        {"type": "text", "text": SUFFIX},
    ]
