"""msclap inputs must be prepared like the official CLAPWrapper.

The msclap model is replaced by stubs, so nothing is downloaded.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from datasets import Dataset
from torch.utils.data import DataLoader

from mteb.models.model_implementations.msclap_models import MSClapWrapper


class _Tokenizer:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def __call__(self, texts: list[str], **kwargs: Any) -> dict:
        self.texts.extend(texts)
        return {"input_ids": torch.zeros(len(texts), 4, dtype=torch.long)}


def _wrapper(text_model: str) -> MSClapWrapper:
    wrapper = MSClapWrapper.__new__(MSClapWrapper)
    wrapper.device = "cpu"
    wrapper.text_length = 77
    wrapper.tokenizer = _Tokenizer()
    wrapper.model = SimpleNamespace(
        args=SimpleNamespace(text_model=text_model),
        clap=SimpleNamespace(
            caption_encoder=lambda features: torch.ones(
                features["input_ids"].shape[0], 8
            )
        ),
    )
    return wrapper


@pytest.mark.parametrize(
    ("text_model", "suffix"),
    [
        ("gpt2", " <|endoftext|>"),  # msclap-2023
        ("bert-base-uncased", ""),  # msclap-2022
    ],
)
def test_text_gets_official_suffix(text_model, suffix) -> None:
    wrapper = _wrapper(text_model)
    loader = DataLoader(Dataset.from_dict({"text": ["a dog barks", "rain"]}))

    wrapper.get_text_embeddings(loader, show_progress_bar=False)

    assert wrapper.tokenizer.texts == ["a dog barks" + suffix, "rain" + suffix]
