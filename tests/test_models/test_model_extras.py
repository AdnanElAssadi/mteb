from __future__ import annotations

from pathlib import Path

import pytest
from packaging.requirements import Requirement

import mteb

tomllib = pytest.importorskip("tomllib")  # Python 3.11+

_PYPROJECT = Path(__file__).parents[2] / "pyproject.toml"


def _extras_packages(groups: list[str]) -> set[str]:
    with _PYPROJECT.open("rb") as f:
        extras = tomllib.load(f)["project"]["optional-dependencies"]
    extras = {name.replace("_", "-").lower(): reqs for name, reqs in extras.items()}
    return {
        Requirement(req).name.lower()
        for group in groups
        for req in extras[group.replace("_", "-").lower()]
    }


@pytest.mark.parametrize(
    ("model_name", "packages"),
    [
        ("microsoft/speecht5_asr", {"sentencepiece"}),
        ("microsoft/speecht5_tts", {"sentencepiece"}),
        ("microsoft/speecht5_multimodal", {"sentencepiece"}),
        ("facebook/seamless-m4t-v2-large", {"sentencepiece", "protobuf"}),
        ("lyrebird/wav2clip", {"accelerate"}),
        ("nvidia/audio-flamingo-3-hf", {"accelerate"}),
        ("BidirLM/BidirLM-Omni-2.5B-Embedding", {"librosa"}),
        ("nielsr/imagebind-huge", {"einops", "setuptools"}),
        (
            "OpenGVLab/ViCLIP-L-14-hf",
            {"einops", "timm", "ftfy", "regex", "fvcore", "setuptools"},
        ),
        (
            "OpenGVLab/ViCLIP-B-16-hf",
            {"einops", "timm", "ftfy", "regex", "fvcore", "setuptools"},
        ),
    ],
)
def test_model_extras_declare_loader_imports(model_name: str, packages: set[str]):
    """Packages the loader (or the model's remote code) imports must be in its extras."""
    meta = mteb.get_model_meta(model_name)
    assert packages <= _extras_packages(meta._resolve_extras_groups())
