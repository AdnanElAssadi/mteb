"""YAMNet must compute log-mel patches like the official code.

Patches: 0.96 s, 0.48 s hop, padded. Log-mel: yamnet/features.py (tf.signal.stft
without centring, tf.signal.linear_to_mel_weight_matrix).

Uses randomly initialised weights, so nothing is downloaded.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from mteb._create_dataloaders import _custom_collate_fn

pytest.importorskip("torch_vggish_yamnet", reason="torch-vggish-yamnet not installed")

from torch_vggish_yamnet import yamnet

from mteb.models.model_implementations.yamnet_models import yamnet_loader

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
def model(monkeypatch):
    random_yamnet = yamnet.yamnet

    def build(pretrained=True):
        torch.manual_seed(0)
        return random_yamnet(pretrained=False)

    monkeypatch.setattr(yamnet, "yamnet", build)
    return yamnet_loader(device="cpu")


# official yamnet (features.pad_waveform + frame) patch counts
@pytest.mark.parametrize(("seconds", "n_patches"), [(0.3, 1), (1.5, 3), (2.5, 5)])
def test_official_patching(model, seconds, n_patches) -> None:
    seen = []
    model.model.register_forward_pre_hook(lambda _, args: seen.append(args[0].clone()))
    rng = np.random.default_rng(0)
    clip = (rng.standard_normal(int(seconds * SAMPLING_RATE)) * 0.1).astype(np.float32)

    loader = DataLoader(
        _AudioDataset([clip]), batch_size=1, collate_fn=_custom_collate_fn
    )
    embedding = np.asarray(model.get_audio_embeddings(loader, show_progress_bar=False))

    patches = seen[0]
    assert patches.shape == (n_patches, 1, 96, 64)
    # each patch is a 0.96 s slice of the log-mel, 0.48 s after the previous one
    padded = np.zeros(15_600 + (n_patches - 1) * 7_680, dtype=np.float32)
    padded[: len(clip)] = clip
    log_mel = model._log_mel(torch.from_numpy(padded))
    for i in range(n_patches):
        torch.testing.assert_close(patches[i, 0], log_mel[48 * i : 48 * i + 96])

    with torch.no_grad():
        per_patch, _ = model.model(patches)
    expected = per_patch.reshape(n_patches, -1).mean(0).numpy()
    np.testing.assert_allclose(embedding[0], expected, rtol=1e-5, atol=1e-6)


# official yamnet/features.py on a 0.6 s chirp (padded with features.pad_waveform):
# patch 0, frames [0, 1, 30, 59, 95], every 9th mel band, computed with TensorFlow
OFFICIAL_LOG_MEL = np.array(
    [
        [0.8688, -1.0008, -5.0579, -6.2488, -6.6913, -6.8225, -6.8728, -6.8981],
        [-0.8323, 0.7129, -4.7237, -6.2212, -6.5993, -6.7779, -6.8534, -6.8936],
        [-6.8454, -6.6687, -6.2618, -4.2097, -1.1577, -6.1468, -6.7592, -6.8631],
        [-1.4553, -1.0899, -0.7365, -0.2983, 0.3982, 2.2409, 1.1201, 0.6718],
        [-6.9078, -6.9078, -6.9078, -6.9078, -6.9078, -6.9078, -6.9078, -6.9078],
    ],
    dtype=np.float32,
)


def test_official_log_mel(model) -> None:
    seen = []
    model.model.register_forward_pre_hook(lambda _, args: seen.append(args[0].clone()))
    t = np.arange(9_600) / SAMPLING_RATE
    clip = (0.5 * np.sin(2 * np.pi * (200 * t + 3000 * t**2))).astype(np.float32)

    loader = DataLoader(
        _AudioDataset([clip]), batch_size=1, collate_fn=_custom_collate_fn
    )
    model.get_audio_embeddings(loader, show_progress_bar=False)

    patch = seen[0][0, 0].numpy()
    np.testing.assert_allclose(
        patch[[0, 1, 30, 59, 95]][:, ::9], OFFICIAL_LOG_MEL, atol=2e-3
    )
