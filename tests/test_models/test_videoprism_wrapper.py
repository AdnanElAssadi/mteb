from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from datasets import Dataset
from torch.utils.data import DataLoader

from mteb.mocks.mock_tasks import MockRetrievalTask
from mteb.models.model_implementations.videoprism_models import (
    MAX_TEXT_TOKENS,
    VideoPrismClipWrapper,
)

transformers = pytest.importorskip("transformers")


class _RecordingTokenizer:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def __call__(self, texts: list[str], **kwargs: Any) -> object:
        assert kwargs["max_length"] == MAX_TEXT_TOKENS
        self.texts.extend(texts)
        ids = torch.zeros(len(texts), MAX_TEXT_TOKENS, dtype=torch.long)
        return transformers.BatchEncoding({"input_ids": ids})


class _FakeClipModel(torch.nn.Module):
    def get_text_features(self, input_ids: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(pooler_output=torch.zeros(len(input_ids), 4))


@pytest.fixture
def wrapper(monkeypatch: pytest.MonkeyPatch) -> VideoPrismClipWrapper:
    processor = SimpleNamespace(tokenizer=_RecordingTokenizer())
    monkeypatch.setattr(
        transformers.AutoProcessor,
        "from_pretrained",
        staticmethod(lambda *args, **kwargs: processor),
    )
    monkeypatch.setattr(
        transformers.VideoPrismClipModel,
        "from_pretrained",
        staticmethod(lambda *args, **kwargs: _FakeClipModel()),
    )
    return VideoPrismClipWrapper(
        "google/videoprism-lvt-base-f16r288", revision=None, device="cpu"
    )


def test_text_is_canonicalized_like_official_tokenize_texts(
    wrapper: VideoPrismClipWrapper,
) -> None:
    """videoprism.utils.tokenize_texts canonicalises text by default."""
    texts = ["Hello, World!", "Hello,World..", "  Hello   WORLD", "A man's dog-walk"]
    wrapper.encode(
        DataLoader(Dataset.from_dict({"text": texts}), batch_size=2),
        task_metadata=MockRetrievalTask.metadata,
        hf_split="test",
        hf_subset="default",
        show_progress_bar=False,
    )
    assert wrapper.processor.tokenizer.texts == [
        "hello world.",
        "hello world.",
        "hello world.",
        "a man s dog walk.",
    ]
