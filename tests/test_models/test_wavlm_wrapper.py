"""WavLM embeddings must pool ``last_hidden_state``.

On transformers 5, ``hidden_states[-1]`` of a stable-layer-norm checkpoint
(wavlm-large) is taken before the encoder's final LayerNorm, so it differs
from ``last_hidden_state``. On transformers 4 the two were identical.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from mteb._create_dataloaders import _custom_collate_fn

pytest.importorskip("torchaudio", reason="Audio dependencies are not installed")

from transformers import Wav2Vec2FeatureExtractor, WavLMConfig, WavLMModel

from mteb.models.model_implementations.wavlm_models import WavlmWrapper

SAMPLING_RATE = 16_000


class _AudioDataset(Dataset):
    features = {"audio": None}

    def __init__(self, clips: list[np.ndarray]) -> None:
        self.clips = clips

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, i: int) -> dict:
        return {"audio": {"array": self.clips[i], "sampling_rate": SAMPLING_RATE}}


@pytest.fixture
def tiny_stable_layer_norm_wavlm(tmp_path) -> str:
    torch.manual_seed(0)
    config = WavLMConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=37,
        conv_dim=(32, 32, 32),
        conv_stride=(4, 4, 4),
        conv_kernel=(8, 8, 8),
        num_conv_pos_embeddings=16,
        num_conv_pos_embedding_groups=2,
        # wavlm-large layout
        feat_extract_norm="layer",
        do_stable_layer_norm=True,
    )
    model = WavLMModel(config)
    # a non-trivial final LayerNorm, so skipping it is visible
    torch.nn.init.normal_(model.encoder.layer_norm.weight, mean=1.0, std=0.5)
    torch.nn.init.normal_(model.encoder.layer_norm.bias, std=0.5)
    model.save_pretrained(tmp_path)
    Wav2Vec2FeatureExtractor(
        sampling_rate=SAMPLING_RATE, return_attention_mask=True
    ).save_pretrained(tmp_path)
    return str(tmp_path)


def test_pools_last_hidden_state(tiny_stable_layer_norm_wavlm) -> None:
    model = WavlmWrapper(model_name=tiny_stable_layer_norm_wavlm, device="cpu")
    clip = np.random.default_rng(0).standard_normal(SAMPLING_RATE) * 0.1
    clip = clip.astype(np.float32)

    loader = DataLoader(
        _AudioDataset([clip]), batch_size=1, collate_fn=_custom_collate_fn
    )
    embedding = np.asarray(model.get_audio_embeddings(loader, show_progress_bar=False))

    features = model.feature_extractor(
        [clip], sampling_rate=SAMPLING_RATE, return_tensors="pt"
    )
    with torch.no_grad():
        expected = model.model(features.input_values).last_hidden_state.mean(1)

    np.testing.assert_allclose(embedding, expected.numpy(), rtol=1e-4, atol=1e-5)
