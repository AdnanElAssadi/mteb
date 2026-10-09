"""The HF CLAP wrapper must prepare inputs the way laion_clap does.

laion_clap rounds audio to 16-bit before its log-mel and cuts captions at 77
tokens; the larger_clap checkpoints are bit-identical to laion_clap ones and
only match them with both steps.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from mteb._create_dataloaders import _custom_collate_fn

pytest.importorskip("torchaudio", reason="Audio dependencies are not installed")

from mteb.models.model_implementations.clap_models import ClapZeroShotWrapper

SAMPLING_RATE = 48_000


class _Dataset(Dataset):
    def __init__(self, column: str, items: list) -> None:
        self.features = {column: None}
        self.column = column
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        return {self.column: self.items[i]}


class _RecordingProcessor:
    def __init__(self) -> None:
        self.feature_extractor = SimpleNamespace(
            truncation="rand_trunc", nb_max_samples=480_000
        )
        self.calls: list[dict] = []

    def __call__(self, **kwargs: Any) -> dict[str, torch.Tensor]:
        self.calls.append(kwargs)
        if "audio" in kwargs:
            return {"input_features": torch.zeros(1, 1, 4, 4)}
        n = len(kwargs["text"])
        return {
            "input_ids": torch.zeros(n, 3, dtype=torch.long),
            "attention_mask": torch.ones(n, 3, dtype=torch.long),
        }


class _FakeModel:
    def get_audio_features(
        self, input_features: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        return torch.ones(input_features.shape[0], 4)

    def get_text_features(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return torch.ones(input_ids.shape[0], 4)


def _wrapper() -> ClapZeroShotWrapper:
    wrapper = ClapZeroShotWrapper.__new__(ClapZeroShotWrapper)
    wrapper.device = "cpu"
    wrapper.model = _FakeModel()
    wrapper.processor = _RecordingProcessor()
    wrapper.sampling_rate = SAMPLING_RATE
    return wrapper


def test_clap_audio_is_rounded_to_16_bit_like_laion_clap():
    rng = np.random.default_rng(0)
    clip = (rng.standard_normal(SAMPLING_RATE) * 0.1).astype(np.float32)
    clip[:1000] = 1e-6  # near-silence below one 16-bit step becomes exact zeros
    clip[1000] = 1.5  # clipped as laion_clap does
    wrapper = _wrapper()
    loader = DataLoader(
        _Dataset("audio", [{"array": clip, "sampling_rate": SAMPLING_RATE}]),
        batch_size=1,
        collate_fn=_custom_collate_fn,
    )
    wrapper.get_audio_embeddings(loader, show_progress_bar=False)

    seen = np.asarray(wrapper.processor.calls[0]["audio"])
    # laion_clap.training.data: int16_to_float32(float32_to_int16(x))
    expected = (np.clip(clip, -1.0, 1.0) * 32767.0).astype(np.int16) / 32767.0
    np.testing.assert_allclose(seen, expected.astype(np.float32), rtol=0, atol=0)
    assert not seen[:1000].any()


def test_clap_text_is_cut_at_laion_clap_context_length():
    wrapper = _wrapper()
    loader = DataLoader(
        _Dataset("text", ["a dog barks", "rain"]),
        batch_size=2,
        collate_fn=_custom_collate_fn,
    )
    wrapper.get_text_embeddings(loader, show_progress_bar=False)

    call = wrapper.processor.calls[0]
    assert call["truncation"] is True
    assert call["max_length"] == 77
