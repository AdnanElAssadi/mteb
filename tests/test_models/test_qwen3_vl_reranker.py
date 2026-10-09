from __future__ import annotations

import json
from pathlib import Path

import pytest

import mteb
from mteb.models.model_implementations import qwen3_vl_reranker
from mteb.models.model_implementations.qwen3_vl_reranker import Qwen3VLRerankerWrapper
from mteb.models.sentence_transformer_wrapper import CrossEncoderWrapper


def test_8b_loads_score_head_from_shipped_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Layout of Qwen/Qwen3-VL-Reranker-8B at the pinned revision: modules.json
    # names "1_CausalScoreHead", but the folder in the repo is "1_LogitScore".
    snapshot = tmp_path / "snapshot"
    (snapshot / "1_LogitScore").mkdir(parents=True)
    (snapshot / "1_LogitScore" / "config.json").write_text('{"true_token_id": 1}')
    (snapshot / "modules.json").write_text(
        json.dumps(
            [
                {"idx": 0, "name": "0", "path": "", "type": "Transformer"},
                {"idx": 1, "name": "1", "path": "1_CausalScoreHead", "type": "Logit"},
            ]
        )
    )
    requested = {}

    def fake_snapshot_download(repo_id: str, revision: str | None = None) -> str:
        requested.update(repo_id=repo_id, revision=revision)
        return str(snapshot)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)

    meta = mteb.get_model_meta("Qwen/Qwen3-VL-Reranker-8B")
    local = Path(
        qwen3_vl_reranker._snapshot_with_module_paths(
            meta.name, meta.revision, meta.loader_kwargs["module_paths"]
        )
    )
    assert requested == {"repo_id": meta.name, "revision": meta.revision}
    paths = [m["path"] for m in json.loads((local / "modules.json").read_text())]
    assert paths == ["", "1_LogitScore"]
    assert (local / "1_LogitScore" / "config.json").exists()
    # the original snapshot is left untouched
    assert "1_CausalScoreHead" in (snapshot / "modules.json").read_text()


def test_2b_does_not_rewrite_modules() -> None:
    meta = mteb.get_model_meta("Qwen/Qwen3-VL-Reranker-2B")
    assert "module_paths" not in (meta.loader_kwargs or {})


def test_multimodal_inputs_are_ordered_media_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    items = [
        {"text": "t", "image": "i"},
        {"text": "t", "video": "v"},
        {"text": "t", "image": "i", "video": "v"},
        "plain text",
    ]
    monkeypatch.setattr(
        CrossEncoderWrapper, "_collect_inputs", lambda self, loader, prefix: items
    )
    wrapper = Qwen3VLRerankerWrapper.__new__(Qwen3VLRerankerWrapper)
    ordered = wrapper._collect_inputs(None, "")
    assert [list(item) if isinstance(item, dict) else item for item in ordered] == [
        ["image", "text"],
        ["video", "text"],
        ["video", "image", "text"],
        "plain text",
    ]
