"""msclap inputs must be prepared like the official CLAPWrapper.

The msclap model is replaced by stubs, so nothing is downloaded.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from mteb._create_dataloaders import _custom_collate_fn
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


class _Dataset(TorchDataset):
    features = {"audio": None}

    def __init__(self, clips: list[np.ndarray], sampling_rate: int) -> None:
        self.clips = clips
        self.sampling_rate = sampling_rate

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, i: int) -> dict:
        return {"audio": {"array": self.clips[i], "sampling_rate": self.sampling_rate}}


def test_audio_reaches_msclap_without_16_bit_quantisation() -> None:
    sf = pytest.importorskip("soundfile")

    written = []

    def get_audio_embeddings(paths, resample):
        for path in paths:
            written.append((sf.info(path).subtype, sf.read(path, dtype="float32")[0]))
        return torch.ones(len(paths), 8)

    wrapper = _wrapper("gpt2")
    wrapper.sampling_rate = 44_100
    wrapper.max_audio_length_seconds = 7.0
    wrapper.model.get_audio_embeddings = get_audio_embeddings
    clip = np.random.default_rng(0).uniform(-0.5, 0.5, 4_410).astype(np.float32)
    loader = DataLoader(
        _Dataset([clip], 44_100), batch_size=1, collate_fn=_custom_collate_fn
    )

    wrapper.get_audio_embeddings(loader, show_progress_bar=False)

    subtype, audio = written[0]
    assert subtype == "FLOAT"
    np.testing.assert_array_equal(audio, clip)
