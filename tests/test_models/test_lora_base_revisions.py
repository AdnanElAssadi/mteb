"""LoRA adapter wrappers must load their base model at a pinned revision, not at main."""

from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import mteb

ADAPTER_REVISION = "0123456789abcdef0123456789abcdef01234567"
BASE_REVISION = "fedcba9876543210fedcba9876543210fedcba98"


class _Recorder:
    """Stand-in for a colpali-engine class; records from_pretrained kwargs."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def from_pretrained(self, *args, **kwargs) -> MagicMock:  # noqa: ANN002, ANN003
        self.calls.append(kwargs)
        return MagicMock()


@pytest.mark.parametrize(
    "name", ["vidore/colqwen-omni-v0.1", "Tevatron/OmniEmbed-v0.1"]
)
def test_lora_adapter_metas_pin_the_base_model(name: str) -> None:
    base_revision = mteb.get_model_meta(name).loader_kwargs.get("base_revision")
    assert base_revision is not None
    assert re.fullmatch(r"[0-9a-f]{40}", base_revision)


def test_colpali_engine_loads_base_model_at_base_revision() -> None:
    from mteb.models.model_implementations.colpali_models import ColPaliEngineWrapper

    model_class, processor_class = _Recorder(), _Recorder()
    ColPaliEngineWrapper(
        "vidore/colqwen-omni-v0.1",
        model_class=model_class,
        processor_class=processor_class,
        revision=ADAPTER_REVISION,
        base_revision=BASE_REVISION,
        device="cpu",
    )
    (call,) = model_class.calls
    # transformers loads the adapter at adapter_kwargs["revision"] and its base at `revision`
    assert call["adapter_kwargs"] == {"revision": ADAPTER_REVISION}
    assert call["revision"] == BASE_REVISION
    assert "base_revision" not in call


def _fake_snapshots(tmp_path: Path) -> dict[tuple[str, str | None], Path]:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "org/base", "revision": None, "r": 16})
    )
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    (adapter / "1_Pooling").mkdir()
    base = tmp_path / "base"
    base.mkdir()
    return {
        ("Tevatron/OmniEmbed-v0.1", ADAPTER_REVISION): adapter,
        ("org/base", BASE_REVISION): base,
    }


def test_tevatron_loads_base_model_at_base_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import huggingface_hub
    import sentence_transformers

    from mteb.models.model_implementations.tevatron_omni_embed_models import (
        TevatronOmniEmbedWrapper,
    )

    snapshots = _fake_snapshots(tmp_path)
    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda repo_id, revision=None, **_: str(snapshots[(repo_id, revision)]),
    )
    loaded = {}

    def fake_sentence_transformer(model, revision=None, **kwargs):  # noqa: ANN003
        loaded.update(model=model, revision=revision)
        st_model = MagicMock()
        st_model.prompts = {}
        return st_model

    monkeypatch.setattr(
        sentence_transformers, "SentenceTransformer", fake_sentence_transformer
    )

    TevatronOmniEmbedWrapper(
        "Tevatron/OmniEmbed-v0.1",
        revision=ADAPTER_REVISION,
        device="cpu",
        base_revision=BASE_REVISION,
    )

    local = Path(loaded["model"])
    adapter_config = json.loads((local / "adapter_config.json").read_text())
    # the base model is loaded from its snapshot at BASE_REVISION ...
    assert adapter_config["base_model_name_or_path"] == str(
        snapshots[("org/base", BASE_REVISION)]
    )
    assert adapter_config["r"] == 16
    # ... and every other file comes from the adapter snapshot at ADAPTER_REVISION
    adapter = snapshots[("Tevatron/OmniEmbed-v0.1", ADAPTER_REVISION)]
    assert (local / "adapter_model.safetensors").resolve() == (
        adapter / "adapter_model.safetensors"
    ).resolve()
    assert (local / "1_Pooling").resolve() == (adapter / "1_Pooling").resolve()
    assert loaded["revision"] is None
