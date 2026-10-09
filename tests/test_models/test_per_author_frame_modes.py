"""Each video/audio wrapper builds its collator with its authors' frame sampler / stereo handling."""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

import pytest
from torch.utils.data import DataLoader, Dataset


class _BuiltError(Exception):
    pass


class _Dataset(Dataset):
    def __init__(self, feature: str) -> None:
        self.features = {feature: None}

    def __len__(self) -> int:
        return 1

    def __getitem__(self, i: int) -> dict[str, Any]:
        return {}


def _spy(
    monkeypatch: pytest.MonkeyPatch, module: ModuleType, seen: list[dict[str, Any]]
) -> None:
    def _record(**kwargs: Any) -> None:
        seen.append(kwargs)
        raise _BuiltError

    for name in ("FramesCollator", "VideoCollator", "AudioCollator"):
        if hasattr(module, name):
            monkeypatch.setattr(module, name, _record)
    monkeypatch.setattr(
        "mteb.models.modality_collators.VideoCollator", _record, raising=True
    )
    monkeypatch.setattr(
        "mteb.models.modality_collators.AudioCollator", _record, raising=True
    )


def _build(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    class_name: str,
    method: str,
    feature: str = "video",
    **attrs: Any,
) -> dict[str, Any]:
    module = importlib.import_module(f"mteb.models.model_implementations.{module_name}")
    seen: list[dict[str, Any]] = []
    _spy(monkeypatch, module, seen)
    wrapper = object.__new__(getattr(module, class_name))
    defaults = dict(
        fps=None,
        max_frames=None,
        num_frames=8,
        max_samples=None,
        sampling_rate=16_000,
        target_sampling_rate=16_000,
        device="cpu",
    )
    for key, value in {**defaults, **attrs}.items():
        setattr(wrapper, key, value)
    loader = DataLoader(_Dataset(feature), batch_size=1)
    kwargs = (
        dict(task_metadata=None, hf_split="test", hf_subset="default")
        if method == "encode"
        else {}
    )
    with pytest.raises(_BuiltError):
        getattr(wrapper, method)(loader, **kwargs)
    return seen[0]


@pytest.mark.parametrize(
    ("module_name", "class_name", "method", "expected"),
    [
        ("xclip_models", "XCLIPModel", "encode", "segment_middle"),
        ("cosmos_embed_models", "CosmosEmbed1Model", "encode", "linspace_floor"),
        ("videoprism_models", "VideoPrismVisionWrapper", "encode", "uniform"),
        ("videoprism_models", "VideoPrismClipWrapper", "encode", "uniform"),
        ("facebook_jepa", "VJepaV2Wrapper", "encode", "pad_last"),
        (
            "language_bind_models",
            "LanguageBindVideoWrapper",
            "get_video_embeddings",
            "linspace_floor",
        ),
        ("jina_models", "JinaV5OmniWrapper", "encode", "linspace_floor_no_pad"),
    ],
)
def test_wrapper_frame_index_mode(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    class_name: str,
    method: str,
    expected: str,
) -> None:
    kwargs = _build(monkeypatch, module_name, class_name, method)
    assert kwargs["frame_index_mode"] == expected


def test_ebind_uses_one_fps_grid(monkeypatch: pytest.MonkeyPatch) -> None:
    from mteb.models.model_implementations.ebind_models import _ebind_frame_indices

    kwargs = _build(monkeypatch, "ebind_models", "EBindWrapper", "encode")
    assert kwargs["frame_index_mode"] is _ebind_frame_indices
    assert kwargs["downmix"] == "first"
    # 40 s at 30 fps: 8 of the 40 points of the 1 fps grid (transforms.load_video)
    grid = list(range(0, 1200, 30))
    assert _ebind_frame_indices(1200, 8, 30.0) == [
        grid[round(i * 39 / 7)] for i in range(8)
    ]
    # grid shorter than 8 points: 8 frames over the whole clip
    assert _ebind_frame_indices(150, 8, 30.0) == [0, 21, 43, 64, 85, 106, 128, 149]
    # fewer frames than 8: mteb's default (repeat in place)
    assert _ebind_frame_indices(5, 8, 25.0) == [0, 0, 1, 1, 2, 3, 3, 4]


def test_vlm2vec_v2_frame_indices(monkeypatch: pytest.MonkeyPatch) -> None:
    from mteb.models.model_implementations.vlm2vec_models import (
        _vlm2vec_frame_indices,
    )

    kwargs = _build(
        monkeypatch,
        "vlm2vec_models",
        "VLM2VEC2Wrapper",
        "encode",
        model=None,
        processor=None,
    )
    assert kwargs["frame_index_mode"] is _vlm2vec_frame_indices
    # 1200 frames: floor(i * 1200 / 64) saved, then linspace(0, 63, 8, dtype=int)
    saved = [i * 1200 // 64 for i in range(64)]
    assert _vlm2vec_frame_indices(1200, 8, None) == [
        saved[i] for i in [0, 9, 18, 27, 36, 45, 54, 63]
    ]
    assert _vlm2vec_frame_indices(30, 8, None) == [0, 4, 8, 12, 16, 20, 24, 29]
    # no padding for short clips
    assert _vlm2vec_frame_indices(5, 8, None) == [0, 1, 2, 3, 4]


@pytest.mark.parametrize(
    ("class_name", "method", "feature"),
    [
        ("LanguageBindAudioWrapper", "get_audio_embeddings", "audio"),
        ("LanguageBindVideoWrapper", "get_video_embeddings", "video"),
    ],
)
def test_languagebind_keeps_first_channel(
    monkeypatch: pytest.MonkeyPatch, class_name: str, method: str, feature: str
) -> None:
    kwargs = _build(
        monkeypatch, "language_bind_models", class_name, method, feature=feature
    )
    assert kwargs["downmix"] == "first"


def test_ebind_audio_keeps_first_channel(monkeypatch: pytest.MonkeyPatch) -> None:
    kwargs = _build(
        monkeypatch, "ebind_models", "EBindWrapper", "encode", feature="audio"
    )
    assert kwargs["downmix"] == "first"


def test_internvideo2_uses_segment_middle_pad_last(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mteb
    from mteb.models import sentence_transformer_wrapper as st_wrapper

    meta = mteb.get_model_meta("OpenGVLab/InternVideo2-CLIP-1B-224p-f8")
    assert meta.loader_kwargs["frame_index_mode"] == "segment_middle_pad_last"
    seen: list[dict[str, Any]] = []
    _spy(monkeypatch, st_wrapper, seen)
    with pytest.raises(_BuiltError):
        st_wrapper._setup_modality_collator(
            DataLoader(_Dataset("video"), batch_size=1),
            fps=None,
            max_frames=None,
            num_frames=8,
            target_sampling_rate=None,
            max_samples=None,
            frame_index_mode="segment_middle_pad_last",
        )
    assert seen[0]["frame_index_mode"] == "segment_middle_pad_last"
