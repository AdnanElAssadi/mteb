"""SpeechT5 resamples non-16 kHz audio through the shared AudioCollator filter."""

from __future__ import annotations

import numpy as np
import pytest
from torch.utils.data import DataLoader

pytest.importorskip("torchaudio", reason="Audio dependencies are not installed")

from mteb.models.modality_collators import AudioCollator
from mteb.models.model_implementations.speecht5_models import SpeechT5Audio


class _StopEncodingError(Exception):
    pass


@pytest.mark.parametrize("sampling_rate", [44_100, 8_000])
def test_speecht5_resamples_with_collator_default(sampling_rate: int) -> None:
    seen = []

    def processor(*, audio: list[np.ndarray], **kwargs: object) -> None:
        seen.extend(audio)
        raise _StopEncodingError

    # no weights: only the resampling before the processor is exercised
    model = object.__new__(SpeechT5Audio)
    model.sampling_rate = 16_000
    model.max_samples = 16_000 * 10
    model.device = "cpu"
    model.asr_processor = processor

    clip = np.random.default_rng(0).standard_normal(sampling_rate).astype(np.float32)
    inputs = DataLoader(
        [{"audio": {"array": clip, "sampling_rate": sampling_rate}}],
        batch_size=1,
        collate_fn=lambda rows: {"audio": [r["audio"] for r in rows]},
    )

    with pytest.raises(_StopEncodingError):
        model.get_audio_embeddings(inputs, show_progress_bar=False)

    expected = AudioCollator.resample_audio(
        {"audio": {"array": clip, "sampling_rate": sampling_rate}},
        target_sampling_rate=16_000,
    )
    torchaudio_default = AudioCollator.resample_audio(
        {"audio": {"array": clip, "sampling_rate": sampling_rate}},
        target_sampling_rate=16_000,
        resampling="torchaudio_default",
    )
    assert len(seen) == 1
    np.testing.assert_array_equal(seen[0], expected)
    assert not np.allclose(seen[0], torchaudio_default, atol=1e-3)
