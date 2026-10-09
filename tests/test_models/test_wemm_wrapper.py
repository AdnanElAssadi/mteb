from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import sentence_transformers
from tokenizers import Regex, Tokenizer, models, normalizers

from mteb.models.model_implementations.wemm_models import (
    WeMMEncoderWrapper,
    wemm_embedding_2b,
)


def _tokenizer(normalizer: normalizers.Normalizer) -> Tokenizer:
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0}, "[UNK]"))
    tokenizer.normalizer = normalizer
    return tokenizer


class _FakeSentenceTransformer(list):
    """Tokenizer as transformers>=5.18 builds it (Qwen3_5Tokenizer): plain NFC."""

    def __init__(self, model: str, **kwargs: Any) -> None:
        tokenizer = SimpleNamespace(backend_tokenizer=_tokenizer(normalizers.NFC()))
        processor = SimpleNamespace(tokenizer=tokenizer)
        super().__init__([SimpleNamespace(processor=processor)])
        self.prompts: dict[str, str] = {}


def test_wemm_restores_checkpoint_normalizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the checkpoint's tokenizer.json normalizer drops a trailing newline
    checkpoint = _tokenizer(
        normalizers.Sequence([normalizers.Replace(Regex("\n$"), ""), normalizers.NFC()])
    )
    tokenizer_file = tmp_path / "tokenizer.json"
    checkpoint.save(str(tokenizer_file))
    monkeypatch.setattr(
        sentence_transformers, "SentenceTransformer", _FakeSentenceTransformer
    )
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download", lambda *args, **kwargs: str(tokenizer_file)
    )

    wrapper = WeMMEncoderWrapper(
        wemm_embedding_2b.name,
        revision=wemm_embedding_2b.revision,
        **wemm_embedding_2b.loader_kwargs,
    )

    normalizer = wrapper.model[0].processor.tokenizer.backend_tokenizer.normalizer
    # "<|im_end|>\n" + "<embedding>": the newline must be removed, as with the
    # checkpoint's own tokenizer (transformers<=5.17 and the authors' 5.2 pin)
    assert normalizer.normalize_str("<|im_end|>\n") == "<|im_end|>"
