"""VGGish must match the official TF embedding: no final ReLU, log offset 0.01,
log-mel examples as vggish_input.waveform_to_examples.

Uses randomly initialised weights, so nothing is downloaded.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from mteb._create_dataloaders import _custom_collate_fn

pytest.importorskip("torch_vggish_yamnet", reason="torch-vggish-yamnet not installed")

from torch_vggish_yamnet import vggish
from torch_vggish_yamnet.params import CommonParams

from mteb.models.model_implementations.vggish_models import vggish_loader

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
    get_vggish = vggish.get_vggish

    def random_vggish(with_classifier=False, pretrained=True):
        torch.manual_seed(0)
        return get_vggish(with_classifier=with_classifier, pretrained=False)

    monkeypatch.setattr(vggish, "get_vggish", random_vggish)
    return vggish_loader(device="cpu")


def test_official_embedding_and_log_offset(model) -> None:
    seen = []
    model.model.features.register_forward_pre_hook(
        lambda _, args: seen.append(args[0].clone())
    )
    # 1 s of noise followed by 1 s of silence
    clip = np.zeros(2 * SAMPLING_RATE, dtype=np.float32)
    clip[:SAMPLING_RATE] = np.random.default_rng(0).standard_normal(SAMPLING_RATE)
    clip[:SAMPLING_RATE] *= 0.1
    offset_before = CommonParams.LOG_OFFSET

    loader = DataLoader(
        _AudioDataset([clip]), batch_size=1, collate_fn=_custom_collate_fn
    )
    embedding = np.asarray(model.get_audio_embeddings(loader, show_progress_bar=False))

    # silence sits at log(0.01), VGGish's offset, not at YAMNet's log(0.001)
    assert seen[0].min().item() == pytest.approx(math.log(0.01), abs=1e-4)
    # the shared constant is left alone for YAMNet
    assert offset_before == CommonParams.LOG_OFFSET
    # the embedding layer is linear, so values are not clipped at 0
    assert not isinstance(model.model.embeddings[-1], torch.nn.ReLU)
    with torch.no_grad():
        per_patch = model.model(seen[0])
    assert (per_patch < 0).any()
    np.testing.assert_allclose(embedding[0], per_patch.mean(0).numpy(), rtol=1e-5)


# official vggish_input.waveform_to_examples on a 0.6 s chirp zero-padded to 15,600
# samples (unpadded it returns no example): example 0, frames [0, 1, 30, 59, 95],
# every 9th mel band
OFFICIAL_LOG_MEL = np.array(
    [
        [0.8725, -0.9766, -4.176, -4.5159, -4.5813, -4.5964, -4.6016, -4.6043],
        [-0.8119, 0.7173, -4.0239, -4.511, -4.5696, -4.5914, -4.5996, -4.6038],
        [-4.5988, -4.5785, -4.5183, -3.7359, -1.1295, -4.4972, -4.5892, -4.6006],
        [-1.4175, -1.0635, -0.7178, -0.2862, 0.4042, 2.2419, 1.123, 0.6764],
        [-4.6052, -4.6052, -4.6052, -4.6052, -4.6052, -4.6052, -4.6052, -4.6052],
    ],
    dtype=np.float32,
)


def _examples_seen(model, clip: np.ndarray) -> torch.Tensor:
    seen = []
    model.model.features.register_forward_pre_hook(
        lambda _, args: seen.append(args[0].clone())
    )
    loader = DataLoader(
        _AudioDataset([clip]), batch_size=1, collate_fn=_custom_collate_fn
    )
    model.get_audio_embeddings(loader, show_progress_bar=False)
    return seen[0]


def test_official_log_mel(model) -> None:
    t = np.arange(9_600) / SAMPLING_RATE
    clip = (0.5 * np.sin(2 * np.pi * (200 * t + 3000 * t**2))).astype(np.float32)
    examples = _examples_seen(model, clip)
    assert examples.shape == (1, 1, 96, 64)
    np.testing.assert_allclose(
        examples[0, 0].numpy()[[0, 1, 30, 59, 95]][:, ::9],
        OFFICIAL_LOG_MEL,
        atol=1e-3,
    )


# official example counts (non-centred 25 ms frames, 0.96 s examples, no tail)
@pytest.mark.parametrize(("n_samples", "n_examples"), [(30_720, 1), (30_960, 2)])
def test_official_example_count(model, n_samples, n_examples) -> None:
    clip = np.full(n_samples, 0.1, dtype=np.float32)
    assert _examples_seen(model, clip).shape[0] == n_examples
