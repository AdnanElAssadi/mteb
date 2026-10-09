"""Wrappers whose authors resample with torchaudio's default filter (or with FFmpeg,
which it matches closely) keep it instead of the collator's kaiser_best default."""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any

import pytest


class _BuiltError(Exception):
    """Raised by the fake collators once the wrapper has configured them."""


def _recorder(seen: list[dict[str, Any]]) -> type:
    class _Collator:
        def __init__(self, **kwargs: Any) -> None:
            seen.append(kwargs)
            raise _BuiltError

    return _Collator


def _loader(*features: str) -> SimpleNamespace:
    return SimpleNamespace(
        dataset=SimpleNamespace(features=dict.fromkeys(features)), collate_fn=None
    )


def test_audio_flamingo_resamples_with_torchaudio_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mteb.models.model_implementations import audio_flamingo
    from mteb.models.model_implementations.audio_flamingo import AudioFlamingoWrapper

    seen: list[dict[str, Any]] = []

    def _resample_audio(audio: dict[str, Any], **kwargs: Any) -> None:
        seen.append(kwargs)
        raise _BuiltError

    monkeypatch.setattr(
        audio_flamingo.AudioCollator, "resample_audio", staticmethod(_resample_audio)
    )
    wrapper = object.__new__(AudioFlamingoWrapper)
    wrapper.sampling_rate = 16_000
    wrapper.max_audio_length_seconds = 30.0
    batch = {"audio": [{"array": [0.0], "sampling_rate": 44_100}]}
    with pytest.raises(_BuiltError):
        wrapper.encode(
            [batch],  # type: ignore[arg-type]
            task_metadata=None,  # type: ignore[arg-type]
            hf_split="test",
            hf_subset="default",
            show_progress_bar=False,
        )
    assert seen[0]["resampling"] == "torchaudio_default"


def test_voiceclap_large_v2_and_aurola_resample_with_torchaudio_default() -> None:
    import mteb
    from mteb.models.model_implementations.aurola_omni_models import (
        AuroLAOmniWrapper,
    )

    meta = mteb.get_model_meta("laion/voiceclap-large-v2")
    assert meta.loader_kwargs["audio_resampling"] == "torchaudio_default"
    default = inspect.signature(AuroLAOmniWrapper).parameters["audio_resampling"]
    assert default.default == "torchaudio_default"


def test_languagebind_audio_resamples_with_torchaudio_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mteb.models.model_implementations import language_bind_models as lb

    seen: list[dict[str, Any]] = []
    monkeypatch.setattr(lb, "AudioCollator", _recorder(seen))
    wrapper = object.__new__(lb.LanguageBindAudioWrapper)
    wrapper.sampling_rate = 16_000
    with pytest.raises(_BuiltError):
        wrapper.get_audio_embeddings(_loader("audio"))  # type: ignore[arg-type]
    assert seen[0]["resampling"] == "torchaudio_default"


@pytest.mark.parametrize("feature", ["audio", "video"])
def test_ebind_resamples_with_torchaudio_default(
    monkeypatch: pytest.MonkeyPatch, feature: str
) -> None:
    from mteb.models.model_implementations import ebind_models

    seen: list[dict[str, Any]] = []
    monkeypatch.setattr(ebind_models, "AudioCollator", _recorder(seen))
    monkeypatch.setattr(ebind_models, "VideoCollator", _recorder(seen))
    wrapper = object.__new__(ebind_models.EBindWrapper)
    wrapper.fps, wrapper.max_frames, wrapper.num_frames = None, None, 8
    with pytest.raises(_BuiltError):
        wrapper.encode(
            _loader(feature),  # type: ignore[arg-type]
            task_metadata=None,  # type: ignore[arg-type]
            hf_split="test",
            hf_subset="default",
        )
    assert seen[0]["resampling"] == "torchaudio_default"
