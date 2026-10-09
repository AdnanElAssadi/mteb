from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import sentence_transformers

from mteb.models.model_implementations.aurola_omni_models import (
    AuroLAOmniWrapper,
    aurola_omni_3b,
    aurola_omni_7b,
)


class _FakeSentenceTransformer(list):
    """Records the load kwargs; exposes the one module the wrapper configures."""

    def __init__(self, model: str, **kwargs: Any) -> None:
        processor = SimpleNamespace(
            feature_extractor=SimpleNamespace(sampling_rate=16000)
        )
        super().__init__(
            [
                SimpleNamespace(
                    processor=processor,
                    processing_kwargs={},
                    preprocess=lambda *args, **kwargs: {},
                )
            ]
        )
        self.load_kwargs = kwargs
        self.prompts: dict[str, str] = {}


@pytest.mark.parametrize("meta", [aurola_omni_3b, aurola_omni_7b])
def test_aurola_caps_only_videos(meta, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sentence_transformers, "SentenceTransformer", _FakeSentenceTransformer
    )
    wrapper = AuroLAOmniWrapper(meta.name, revision=meta.revision, **meta.loader_kwargs)

    # processor_kwargs go to AutoProcessor.from_pretrained and resize images too;
    # the model card only caps videos, so images keep the checkpoint's limits
    processor_kwargs = wrapper.model.load_kwargs.get("processor_kwargs") or {}
    assert "max_pixels" not in processor_kwargs
    assert wrapper.model[0].processing_kwargs["video"]["max_pixels"] == 64 * 28 * 28
