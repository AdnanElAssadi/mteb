from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from datasets import Dataset
from PIL import Image
from torch.utils.data import DataLoader

from mteb._create_dataloaders import _custom_collate_fn
from mteb.mocks.mock_tasks import MockRetrievalTask
from mteb.models.model_implementations.lco_embedding_models import LCOEmbedding

SHORTEST_EDGE = 3136


class _FakeTokenizer:
    def __call__(self, text: str) -> SimpleNamespace:
        return SimpleNamespace(input_ids=text.split())


class _FakeInputs(dict):
    def to(self, device: str) -> _FakeInputs:
        return self


class _FakeProcessor:
    def __init__(self) -> None:
        self.tokenizer = _FakeTokenizer()
        self.image_processor = SimpleNamespace(
            size={"shortest_edge": SHORTEST_EDGE, "longest_edge": 12845056}
        )
        self.calls: list[dict[str, Any]] = []

    @staticmethod
    def apply_chat_template(messages: list[dict], **kwargs: Any) -> str:
        return ""

    def __call__(self, text: list[str], **kwargs: Any) -> _FakeInputs:
        self.calls.append(kwargs)
        return _FakeInputs(input_ids=torch.zeros(len(text), 1))


class _FakeModel:
    def __call__(self, input_ids: torch.Tensor, **kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(hidden_states=[torch.ones(len(input_ids), 1, 4)])


def _wrapper() -> LCOEmbedding:
    wrapper = LCOEmbedding.__new__(LCOEmbedding)
    wrapper.processor = _FakeProcessor()
    wrapper.model = _FakeModel()
    wrapper.device = "cpu"
    wrapper.max_image_pixels = 1280 * 28 * 28
    wrapper.max_audio_length = 4_800_000
    return wrapper


def test_lco_image_text_prompt_puts_text_first() -> None:
    image = Image.new("RGB", (8, 8))
    messages = _wrapper()._build_messages(
        {"text": ["a red car"], "image": [image]},
        suffix="\nSummarize the above image in one word:",
    )
    # authors' MIEB fused embeddings: text, image, modality-free suffix
    assert messages[0][0]["content"] == [
        {"type": "text", "text": "a red car"},
        {"type": "image", "image": "placeholder"},
        {"type": "text", "text": "\nSummarize the above in one word:"},
    ]


def test_lco_audio_video_prompt_unchanged() -> None:
    suffix = "\nSummarize the above audio in one word:"
    messages = _wrapper()._build_messages(
        {"audio": [object()], "video": [object()]}, suffix=suffix
    )
    assert messages[0][0]["content"] == [
        {"type": "video", "video": "placeholder"},
        {"type": "audio", "audio": "placeholder"},
        {"type": "text", "text": suffix},
    ]


@pytest.mark.parametrize(
    ("task_type", "expected"),
    [
        (
            "Any2AnyRetrieval",
            {"size": {"shortest_edge": SHORTEST_EDGE, "longest_edge": 1280 * 28 * 28}},
        ),
        # document tasks keep full resolution, as in the authors' eval
        ("DocumentUnderstanding", {}),
    ],
)
def test_lco_image_cap_skips_document_tasks(
    task_type: str, expected: dict[str, Any]
) -> None:
    wrapper = _wrapper()
    task_metadata = MockRetrievalTask.metadata.model_copy(update={"type": task_type})
    loader = DataLoader(
        Dataset.from_dict({"text": ["a red car"]}), collate_fn=_custom_collate_fn
    )
    wrapper.encode(
        loader, task_metadata=task_metadata, hf_split="test", hf_subset="default"
    )
    assert wrapper.processor.calls[0]["images_kwargs"] == expected
