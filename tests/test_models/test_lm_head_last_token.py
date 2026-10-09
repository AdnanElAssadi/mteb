from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import transformers

from mteb.models.model_implementations.lco_embedding_models import LCOEmbedding
from mteb.models.model_implementations.ops_mm_models import OpsMMEmbeddingWrapper
from mteb.models.model_implementations.qwen_omni_lm import QwenOmniWrapper

VOCAB = 50


class _FakeLM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lm_head = torch.nn.Linear(4, VOCAB, bias=False)

    def get_output_embeddings(self) -> torch.nn.Module:
        return self.lm_head


def _fake_processor(*args: Any, **kwargs: Any) -> SimpleNamespace:
    return SimpleNamespace(
        tokenizer=SimpleNamespace(padding_side="right"),
        feature_extractor=SimpleNamespace(sampling_rate=16_000),
    )


def _fake_model(*args: Any, **kwargs: Any) -> _FakeLM:
    return _FakeLM()


@pytest.mark.parametrize(
    ("model_classes", "build"),
    [
        (
            ["Qwen2_5OmniThinkerForConditionalGeneration", "Qwen2_5OmniProcessor"],
            lambda: LCOEmbedding("LCO-Embedding/LCO-Embedding-Omni-3B", device="cpu"),
        ),
        (
            ["Qwen2_5OmniThinkerForConditionalGeneration", "AutoProcessor"],
            lambda: QwenOmniWrapper("Qwen/Qwen2.5-Omni-3B", revision="x", device="cpu"),
        ),
        (
            ["Qwen3OmniMoeThinkerForConditionalGeneration", "AutoProcessor"],
            lambda: QwenOmniWrapper(
                "Qwen/Qwen3-Omni-30B-A3B-Instruct", revision="x", device="cpu"
            ),
        ),
        (
            ["AutoModelForImageTextToText", "AutoProcessor"],
            lambda: OpsMMEmbeddingWrapper(
                "OpenSearch-AI/Ops-MM-embedding-v1-2B", device="cpu"
            ),
        ),
    ],
)
def test_lm_head_scores_only_the_last_position(
    model_classes: list[str],
    build: Callable[[], object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Embedding wrappers use hidden states only; the LM head must not score every token."""
    model_cls, processor_cls = model_classes
    monkeypatch.setattr(
        getattr(transformers, model_cls), "from_pretrained", staticmethod(_fake_model)
    )
    monkeypatch.setattr(
        getattr(transformers, processor_cls),
        "from_pretrained",
        staticmethod(_fake_processor),
    )
    wrapper = build()
    model = getattr(wrapper, "base_model", None) or wrapper.model

    hidden_states = torch.randn(3, 7, 4)
    logits = model.lm_head(hidden_states)
    assert logits.shape == (3, 1, VOCAB)
    torch.testing.assert_close(
        logits[:, 0], hidden_states[:, -1] @ model.lm_head.weight.T
    )
