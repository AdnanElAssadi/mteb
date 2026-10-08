"""HeAR-s1.1 must average 2 s windows with a 2 s hop, as in the author's run.

The model is stubbed, so no weights are downloaded.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from mteb._create_dataloaders import _custom_collate_fn
from mteb.models.model_implementations.hear_s11_models import HeARS11AudioWrapper

SAMPLING_RATE = 16_000


class _AudioDataset(Dataset):
    features = {"audio": None}

    def __init__(self, clips: list[np.ndarray]) -> None:
        self.clips = clips

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, i: int) -> dict:
        return {"audio": {"array": self.clips[i], "sampling_rate": SAMPLING_RATE}}


class _Model:
    """Embeds a window as its first and last sample, and records its inputs."""

    def __init__(self) -> None:
        self.inputs: list[torch.Tensor] = []

    def __call__(
        self, input_values: torch.Tensor, return_dict: bool
    ) -> SimpleNamespace:
        self.inputs.append(input_values.clone())
        return SimpleNamespace(pooler_output=input_values[:, [0, -1]])


def test_mean_over_two_second_windows() -> None:
    wrapper = HeARS11AudioWrapper.__new__(HeARS11AudioWrapper)
    wrapper.device = "cpu"
    wrapper.model = _Model()
    clip = np.arange(5 * SAMPLING_RATE, dtype=np.float32) + 1  # 5 s
    loader = DataLoader(
        _AudioDataset([clip]), batch_size=1, collate_fn=_custom_collate_fn
    )

    embedding = wrapper.get_audio_embeddings(loader, show_progress_bar=False)

    # three windows, encoded one at a time; the last one is zero-padded
    assert [tuple(x.shape) for x in wrapper.model.inputs] == [(1, 32_000)] * 3
    windows = torch.cat(wrapper.model.inputs)
    torch.testing.assert_close(windows[:2].flatten(), torch.from_numpy(clip[:64_000]))
    torch.testing.assert_close(windows[2, :16_000], torch.from_numpy(clip[64_000:]))
    assert not windows[2, 16_000:].any()
    expected = windows[:, [0, -1]].mean(0, keepdim=True).numpy()
    np.testing.assert_array_equal(embedding, expected)
