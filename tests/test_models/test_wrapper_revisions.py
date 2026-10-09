"""Wrappers must load every Hub artifact at the pinned ModelMeta revision."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import mteb

REVISION = "0123456789abcdef0123456789abcdef01234567"


class _Recorder:
    """Stand-in for a transformers class; records from_pretrained kwargs."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def from_pretrained(self, *args, **kwargs) -> MagicMock:  # noqa: ANN002, ANN003
        self.calls.append(kwargs)
        return MagicMock()


def _patch(monkeypatch: pytest.MonkeyPatch, *names: str) -> list[_Recorder]:
    import transformers

    recorders = []
    for name in names:
        recorder = _Recorder()
        monkeypatch.setattr(
            getattr(transformers, name), "from_pretrained", recorder.from_pretrained
        )
        recorders.append(recorder)
    return recorders


def test_lco_loads_model_at_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    from mteb.models.model_implementations.lco_embedding_models import LCOEmbedding

    recorders = _patch(
        monkeypatch,
        "Qwen2_5OmniProcessor",
        "Qwen2_5OmniThinkerForConditionalGeneration",
    )
    LCOEmbedding("LCO-Embedding/LCO-Embedding-Omni-3B", revision=REVISION, device="cpu")
    for recorder in recorders:
        assert [call.get("revision") for call in recorder.calls] == [REVISION]


def test_rzen_loads_everything_at_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    from mteb.models.model_implementations.rzen_embed_model import RzenEmbedWrapper

    recorders = _patch(
        monkeypatch, "AutoConfig", "AutoProcessor", "Qwen2VLForConditionalGeneration"
    )
    RzenEmbedWrapper("qihoo360/RzenEmbed", revision=REVISION, device="cpu")
    calls = [call for recorder in recorders for call in recorder.calls]
    assert len(calls) == 4
    assert all(call.get("revision") == REVISION for call in calls)


def test_colpali_engine_loads_processor_at_revision() -> None:
    from mteb.models.model_implementations.colpali_models import ColPaliEngineWrapper

    model_class, processor_class = _Recorder(), _Recorder()
    ColPaliEngineWrapper(
        "vidore/colqwen-omni-v0.1",
        model_class=model_class,
        processor_class=processor_class,
        revision=REVISION,
        device="cpu",
    )
    assert [call.get("revision") for call in processor_class.calls] == [REVISION]


@pytest.mark.parametrize(
    "model_meta",
    [m for m in mteb.get_model_metas() if m.name.startswith("openai/whisper-")],
    ids=lambda m: m.name,
)
def test_whisper_revision_is_pinned(model_meta: mteb.ModelMeta) -> None:
    assert len(model_meta.revision) == 40
