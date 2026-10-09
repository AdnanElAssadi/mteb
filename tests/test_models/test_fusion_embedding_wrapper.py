from __future__ import annotations

from typing import Any

import pytest
import torch
from datasets import Dataset
from torch.utils.data import DataLoader

from mteb._create_dataloaders import _custom_collate_fn
from mteb.models.model_implementations.fusion_embedding_models import (
    FusionEmbeddingWrapper,
    fusion_embedding_1_2b_preview,
    fusion_embedding_2_2b_preview,
)

# max_text_tokens in both checkpoints' config.json
CHECKPOINT_MAX_TEXT_TOKENS = 512


class _FakeModel:
    def __init__(self) -> None:
        self.kwargs: list[dict[str, Any]] = []

    def embed_text_batch(self, texts: list[str], **kwargs: Any) -> torch.Tensor:
        self.kwargs.append(kwargs)
        return torch.ones(len(texts), 4)


def test_fusion_text_uses_checkpoint_cap() -> None:
    wrapper = FusionEmbeddingWrapper.__new__(FusionEmbeddingWrapper)
    wrapper.model = _FakeModel()
    loader = DataLoader(
        Dataset.from_dict({"text": ["a", "b"]}), collate_fn=_custom_collate_fn
    )
    wrapper.get_text_embeddings(loader, show_progress_bar=False)
    # the remote code falls back to the checkpoint's max_text_tokens (512)
    assert wrapper.model.kwargs[0].get("max_tokens") in {
        None,
        CHECKPOINT_MAX_TEXT_TOKENS,
    }


@pytest.mark.parametrize(
    "meta", [fusion_embedding_1_2b_preview, fusion_embedding_2_2b_preview]
)
def test_fusion_max_tokens_metadata(meta) -> None:
    assert meta.max_tokens == CHECKPOINT_MAX_TEXT_TOKENS
