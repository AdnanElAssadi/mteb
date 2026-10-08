"""ImageBind audio preprocessing.

No weights are loaded: audio is compared with the official
``imagebind.data.load_and_transform_audio_data``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mteb.models.model_implementations.imagebind_models import ImageBindWrapper

SAMPLING_RATE = 16_000


def _wrapper() -> ImageBindWrapper:
    wrapper = ImageBindWrapper.__new__(ImageBindWrapper)
    wrapper.device = "cpu"
    return wrapper


@pytest.mark.parametrize("seconds", [0.3, 0.6, 1.5, 3.0])
def test_audio_matches_official_loader(monkeypatch, seconds) -> None:
    data = pytest.importorskip("imagebind.data")
    rng = np.random.default_rng(0)
    clip = (rng.standard_normal(int(seconds * SAMPLING_RATE)) * 0.1).astype(np.float32)
    # the official loader reads files; hand it the same waveform instead
    monkeypatch.setattr(
        data.torchaudio,
        "load",
        lambda path: (torch.from_numpy(clip)[None], SAMPLING_RATE),
    )

    expected = data.load_and_transform_audio_data(["clip.wav"], "cpu")
    actual = _wrapper()._load_audio([{"array": clip, "sampling_rate": SAMPLING_RATE}])

    # the kaldi fbank itself varies by up to ~0.05 between identical calls
    torch.testing.assert_close(actual, expected, atol=0.1, rtol=0)
