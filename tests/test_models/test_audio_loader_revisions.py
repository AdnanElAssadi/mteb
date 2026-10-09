"""The CNN14, MuQ-MuLan and msclap loaders fetch weights at the pinned ModelMeta revision.

The model packages are replaced by stubs that record how they are called.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

import mteb


def _stub_module(monkeypatch: pytest.MonkeyPatch, name: str, **attrs: Any) -> None:
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)


def _load(name: str) -> None:
    meta = mteb.get_model_meta(name)
    meta.loader(meta.name, revision=meta.revision, device="cpu", **meta.loader_kwargs)


def test_cnn14_fetches_pinned_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    class AudioClassifier:
        @classmethod
        def from_hparams(cls, **kwargs: Any) -> AudioClassifier:
            calls.append(kwargs)
            return cls()

    _stub_module(monkeypatch, "speechbrain")
    _stub_module(monkeypatch, "speechbrain.inference")
    _stub_module(
        monkeypatch,
        "speechbrain.inference.classifiers",
        AudioClassifier=AudioClassifier,
    )
    _stub_module(monkeypatch, "speechbrain.utils")
    _stub_module(monkeypatch, "speechbrain.utils.fetching", FetchConfig=SimpleNamespace)

    _load("speechbrain/cnn14-esc50")

    revision = mteb.get_model_meta("speechbrain/cnn14-esc50").revision
    assert calls[0]["fetch_config"].revision == revision


def test_muq_mulan_fetches_pinned_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    class MuQMuLan:
        @classmethod
        def from_pretrained(cls, name: str, **kwargs: Any) -> MuQMuLan:
            calls.append((name, kwargs))
            return cls()

        def eval(self) -> MuQMuLan:
            return self

        def to(self, device: str) -> MuQMuLan:
            return self

    _stub_module(monkeypatch, "muq", MuQMuLan=MuQMuLan)

    _load("OpenMuQ/MuQ-MuLan-large")

    revision = mteb.get_model_meta("OpenMuQ/MuQ-MuLan-large").revision
    assert calls == [("OpenMuQ/MuQ-MuLan-large", {"revision": revision})]


@pytest.mark.parametrize(
    ("name", "weights"),
    [
        ("microsoft/msclap-2022", "CLAP_weights_2022.pth"),
        ("microsoft/msclap-2023", "CLAP_weights_2023.pth"),
    ],
)
def test_msclap_fetches_pinned_revision(
    monkeypatch: pytest.MonkeyPatch, name: str, weights: str
) -> None:
    downloads: list[tuple[str, str, str | None]] = []
    clap_calls: list[dict[str, Any]] = []

    def hf_hub_download(
        repo_id: str, filename: str, revision: str | None = None
    ) -> str:
        downloads.append((repo_id, filename, revision))
        return f"/cache/{filename}"

    class CLAP:
        model_repo = "microsoft/msclap"
        model_name = {
            "2022": "CLAP_weights_2022.pth",
            "2023": "CLAP_weights_2023.pth",
        }

        def __init__(self, **kwargs: Any) -> None:
            clap_calls.append(kwargs)
            self.clap = SimpleNamespace(to=lambda device: self.clap)
            self.tokenizer = None
            self.args = SimpleNamespace(sampling_rate=44100, duration=7)

    _stub_module(monkeypatch, "msclap", CLAP=CLAP)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", hf_hub_download)

    _load(name)

    revision = mteb.get_model_meta(name).revision
    assert len(revision) == 40  # a commit sha of microsoft/msclap
    assert downloads == [("microsoft/msclap", weights, revision)]
    assert clap_calls[0]["model_fp"] == f"/cache/{weights}"


def test_voiceclap_large_v2_requires_image_extra() -> None:
    # its Qwen2-VL processor imports torchvision and Pillow
    meta = mteb.get_model_meta("laion/voiceclap-large-v2")
    assert "image" in meta._resolve_extras_groups()
