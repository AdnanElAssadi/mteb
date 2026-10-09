from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
from PIL import Image

from mteb.models.model_implementations.omnivinci_models import OmniVinciWrapper

PAD = 0


class _FakeProcessor:
    """Per-item features as the remote VILA processor returns them."""

    pad_token_id = PAD

    @staticmethod
    def apply_chat_template(conversation: list[dict], **kwargs: Any) -> str:
        return " ".join(
            c.get("text") or c["type"] for c in conversation[0]["content"]
        ).strip()

    def __call__(self, prompts: list[str]) -> SimpleNamespace:
        # the remote batched path crashes or drops block_sizes for batches
        assert len(prompts) == 1, "processor called with a batch"
        n_tokens = len(prompts[0].split())
        n_images = prompts[0].split().count("image")
        return SimpleNamespace(
            input_ids=torch.arange(1, n_tokens + 1).unsqueeze(0),
            media={"image": [f"pixels-{n_tokens}"]} if n_images else {},
            media_config={"image": {"block_sizes": [(n_tokens, n_tokens)]}}
            if n_images
            else {},
        )


class _FakeModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        batch, length = kwargs["input_ids"].shape
        return SimpleNamespace(hidden_states=[torch.ones(batch, length, 4)])


def _wrapper() -> OmniVinciWrapper:
    wrapper = OmniVinciWrapper.__new__(OmniVinciWrapper)
    wrapper.processor = _FakeProcessor()
    wrapper.model = _FakeModel()
    wrapper.device = "cpu"
    return wrapper


def test_omnivinci_batches_items_independently() -> None:
    wrapper = _wrapper()
    image = Image.new("RGB", (8, 8))
    wrapper._encode_batch(
        {"text": ["short", "a much longer caption here"], "image": [image, image]}
    )

    call = wrapper.model.calls[0]
    # right padding with a mask, so pad tokens are never attended to
    assert call["input_ids"].tolist() == [[1, 2, 0, 0, 0, 0], [1, 2, 3, 4, 5, 6]]
    assert call["attention_mask"].tolist() == [
        [True, True, False, False, False, False],
        [True] * 6,
    ]
    # every image keeps its own block_sizes
    assert call["media"]["image"] == ["pixels-2", "pixels-6"]
    assert call["media_config"]["image"]["block_sizes"] == [(2, 2), (6, 6)]
