from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from torch.utils.data import DataLoader

import mteb
from mteb._create_dataloaders import _custom_collate_fn
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


class _StubDecoder:
    """Minimal VideoDecoder stand-in: returns the requested indices as frames."""

    def __init__(self, num_frames: int, seconds: float) -> None:
        self.metadata = SimpleNamespace(
            num_frames=num_frames,
            end_stream_seconds=seconds,
            average_fps=num_frames / seconds,
        )

    def get_frames_at(self, indices: list[int]) -> SimpleNamespace:
        return SimpleNamespace(data=list(indices))


class _Rows(list):
    def __init__(self, rows: list[dict[str, Any]], features: list[str]) -> None:
        super().__init__(rows)
        self.features = dict.fromkeys(features)


class _FakeCrossEncoder:
    """Stands in for sentence_transformers.CrossEncoder: records what preprocess gets."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.calls: list[tuple[list[Any], dict[str, Any]]] = []

    def preprocess(self, inputs: list[Any], prompt: str | None = None, **kwargs: Any):
        self.calls.append((list(inputs), kwargs))
        return {}

    def predict(self, pairs: list[Any], batch_size: int = 32, **kwargs: Any):
        for start in range(0, len(pairs), batch_size):
            self.preprocess(pairs[start : start + batch_size], **kwargs)
        return [0.0] * len(pairs)


def test_video_pairs_pass_frame_timing_to_the_processor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sentence_transformers

    monkeypatch.setattr(sentence_transformers, "CrossEncoder", _FakeCrossEncoder)
    meta = mteb.get_model_meta("Qwen/Qwen3-VL-Reranker-2B")
    wrapper = Qwen3VLRerankerWrapper(meta.name, revision=meta.revision)
    wrapper.mteb_model_meta = meta

    def loader(rows: list[dict[str, Any]]) -> DataLoader:
        return DataLoader(
            _Rows(rows, list(rows[0])), batch_size=2, collate_fn=_custom_collate_fn
        )

    def predict(queries: list[dict[str, Any]], docs: list[dict[str, Any]]) -> None:
        wrapper.predict(
            loader(queries),
            loader(docs),
            task_metadata=mteb.get_task("NFCorpus").metadata,
            hf_split="test",
            hf_subset="default",
            batch_size=2,
        )

    # 6 s at 25 fps and 45 s at 30 fps
    predict(
        [{"text": "q1"}, {"text": "q2"}],
        [
            {"video": _StubDecoder(150, seconds=6.0), "text": "d1"},
            {"video": _StubDecoder(1350, seconds=45.0), "text": "d2"},
        ],
    )
    predict([{"text": "q3"}], [{"text": "d3"}])

    (pairs, kwargs), (_, text_kwargs) = wrapper.model.calls
    # frames reach sentence-transformers unwrapped and media first, as before
    assert [list(doc) for _, doc in pairs] == [["video", "text"], ["video", "text"]]
    metadata = kwargs["processing_kwargs"]["video"]["video_metadata"]
    assert pairs[0][1]["video"] == metadata[0]["frames_indices"]
    assert [md["fps"] for md in metadata] == [25.0, 30.0]
    assert [md["duration"] for md in metadata] == [6.0, 45.0]
    # text-only batches are left alone
    assert "processing_kwargs" not in text_kwargs


def test_pair_video_metadata_follows_query_document_order() -> None:
    seen: dict[str, Any] = {}

    def preprocess(
        inputs: list[Any], prompt: str | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        seen.update(inputs=inputs, kwargs=kwargs)
        return {}

    wrapped = qwen3_vl_reranker._pass_pair_video_metadata(preprocess)
    query_md, doc_md = {"fps": 24.0}, {"fps": 30.0}
    wrapped(
        [
            (
                {"video": {"array": [1, 2], "video_metadata": query_md}},
                {"video": {"array": [3, 4], "video_metadata": doc_md}, "text": "d"},
            ),
            ("q", "d"),
        ],
        prompt="p",
        processing_kwargs={"video": {"do_sample_frames": False}},
    )
    assert seen["inputs"] == [
        ({"video": [1, 2]}, {"video": [3, 4], "text": "d"}),
        ("q", "d"),
    ]
    assert seen["kwargs"]["processing_kwargs"]["video"] == {
        "do_sample_frames": False,
        "video_metadata": [query_md, doc_md],
    }
