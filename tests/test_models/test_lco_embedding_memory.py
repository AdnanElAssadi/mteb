from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
import transformers

from mteb.models.model_implementations.lco_embedding_models import LCOEmbedding

VOCAB = 50


class _FakeThinker(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lm_head = torch.nn.Linear(4, VOCAB, bias=False)


def test_lco_scores_only_the_last_position(monkeypatch: pytest.MonkeyPatch) -> None:
    """The embedding only needs hidden states; the LM head must not score every token."""
    processor = SimpleNamespace(
        tokenizer=SimpleNamespace(padding_side="right"),
        feature_extractor=SimpleNamespace(sampling_rate=16_000),
    )

    def _processor(*args: Any, **kwargs: Any) -> SimpleNamespace:
        return processor

    def _thinker(*args: Any, **kwargs: Any) -> _FakeThinker:
        return _FakeThinker()

    monkeypatch.setattr(
        transformers.Qwen2_5OmniProcessor, "from_pretrained", staticmethod(_processor)
    )
    monkeypatch.setattr(
        transformers.Qwen2_5OmniThinkerForConditionalGeneration,
        "from_pretrained",
        staticmethod(_thinker),
    )
    wrapper = LCOEmbedding("LCO-Embedding/LCO-Embedding-Omni-3B", device="cpu")

    hidden_states = torch.randn(3, 7, 4)
    logits = wrapper.model.lm_head(hidden_states)
    assert logits.shape == (3, 1, VOCAB)
    torch.testing.assert_close(
        logits[:, 0], hidden_states[:, -1] @ wrapper.model.lm_head.weight.T
    )
