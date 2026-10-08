"""Encodec must not pad sub-second clips to 1 s before the time mean.

Uses a tiny randomly initialised checkpoint saved locally.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from mteb._create_dataloaders import _custom_collate_fn

pytest.importorskip("torchaudio", reason="Audio dependencies are not installed")

from transformers import EncodecConfig, EncodecFeatureExtractor, EncodecModel

from mteb.models.model_implementations.encodec_model import EncodecWrapper

SAMPLING_RATE = 24_000


class _AudioDataset(Dataset):
    features = {"audio": None}

    def __init__(self, clips: list[np.ndarray]) -> None:
        self.clips = clips

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, i: int) -> dict:
        return {"audio": {"array": self.clips[i], "sampling_rate": SAMPLING_RATE}}


@pytest.fixture
def tiny_encodec(tmp_path) -> str:
    torch.manual_seed(0)
    config = EncodecConfig(num_filters=4, hidden_size=16, codebook_dim=16)
    EncodecModel(config).save_pretrained(tmp_path)
    EncodecFeatureExtractor(sampling_rate=SAMPLING_RATE).save_pretrained(tmp_path)
    return str(tmp_path)


@pytest.mark.parametrize("n_samples", [100, 2_400, 14_400])
def test_short_clips_are_not_padded_to_one_second(tiny_encodec, n_samples) -> None:
    model = EncodecWrapper(model_name=tiny_encodec, revision=None, device="cpu")
    clip = np.random.default_rng(0).standard_normal(n_samples).astype(np.float32)
    clip *= 0.1

    loader = DataLoader(
        _AudioDataset([clip]), batch_size=1, collate_fn=_custom_collate_fn
    )
    embedding = np.asarray(model.get_audio_embeddings(loader, show_progress_bar=False))

    # the encoder as the official package runs it; only clips under one
    # 320-sample latent frame are padded to it
    padded = np.pad(clip, (0, max(0, 320 - n_samples)))
    with torch.no_grad():
        latent = model.model.encoder(torch.from_numpy(padded)[None, None])
    expected = torch.nn.functional.normalize(latent.mean(dim=2), dim=-1)

    np.testing.assert_allclose(embedding, expected.numpy(), rtol=1e-4, atol=1e-5)
