"""EBind must hand short audio to its processor unpadded.

The processor is stubbed, so neither ebind nor weights are needed.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mteb.models.model_implementations.ebind_models import EBindWrapper

SAMPLING_RATE = 16_000


@pytest.mark.parametrize(
    ("n_samples", "expected"),
    [
        (160, 400),  # under one 25 ms fbank window: padded to it
        (1_600, 1_600),  # 0.1 s: the processor pads the fbank itself
        (32_000, 32_000),
    ],
)
def test_short_audio_padding(n_samples, expected) -> None:
    sf = pytest.importorskip("soundfile")
    lengths = []

    def audio_processor(path: str) -> torch.Tensor:
        lengths.append(sf.info(path).frames)
        return torch.zeros(1, 128, 204)

    wrapper = EBindWrapper.__new__(EBindWrapper)
    wrapper.device = "cpu"
    wrapper.processor = SimpleNamespace(processors={"audio": audio_processor})
    clip = np.random.default_rng(0).uniform(-0.5, 0.5, n_samples).astype(np.float32)

    wrapper._process_audio_batch([{"array": clip, "sampling_rate": SAMPLING_RATE}])

    assert lengths == [expected]
