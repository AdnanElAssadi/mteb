from __future__ import annotations

import base64
from types import SimpleNamespace

import pytest

import mteb
from mteb.models.model_implementations import bedrock_models
from mteb.models.model_implementations.bedrock_models import (
    NovaMultimodalEmbeddingsModel,
)
from mteb.types import PromptType

pytest.importorskip("botocore", reason="boto3 is not installed")

NOVA_NAME = "bedrock/amazon-nova-2-multimodal-embeddings"


@pytest.fixture
def nova(monkeypatch: pytest.MonkeyPatch) -> NovaMultimodalEmbeddingsModel:
    monkeypatch.setattr(
        bedrock_models, "get_bedrock_runtime_client", lambda *a, **k: object()
    )
    return mteb.get_model(NOVA_NAME)


def test_nova_loads_through_get_model(nova: NovaMultimodalEmbeddingsModel) -> None:
    assert isinstance(nova, NovaMultimodalEmbeddingsModel)
    assert nova._model_id == "amazon.nova-2-multimodal-embeddings-v1:0"


@pytest.mark.parametrize(
    ("task_name", "prompt_type", "expected"),
    [
        ("Banking77Classification", PromptType.query, "CLASSIFICATION"),
        ("CREMA_D", None, "CLASSIFICATION"),
        ("CREMA_DClustering", None, "CLUSTERING"),
        ("ESC50PairClassification", None, "CLASSIFICATION"),
        ("MSRVTTT2V", PromptType.query, "VIDEO_RETRIEVAL"),
        ("MSRVTTT2V", PromptType.document, "GENERIC_INDEX"),
        ("NFCorpus", PromptType.query, "TEXT_RETRIEVAL"),
    ],
)
def test_nova_purpose(
    task_name: str, prompt_type: PromptType | None, expected: str
) -> None:
    metadata = mteb.get_task(task_name).metadata
    assert NovaMultimodalEmbeddingsModel._purpose(metadata, prompt_type) == expected


def test_nova_text_is_capped_at_8192_chars(
    nova: NovaMultimodalEmbeddingsModel,
) -> None:
    payload = nova._text_payload("a" * 45_000, "GENERIC_INDEX")
    assert len(payload["singleEmbeddingParams"]["text"]["value"]) == 8192


def test_nova_video_is_h264_and_fits_inline_limit(
    nova: NovaMultimodalEmbeddingsModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    try:
        from torchcodec.decoders import VideoDecoder
    except Exception:  # torchcodec missing or no FFmpeg libraries
        pytest.skip("torchcodec with FFmpeg is not available")

    # A moving gradient; pure noise trips some H.264 encoders' rate control.
    ramp = torch.arange(128)
    shift = 4 * torch.arange(8).view(8, 1, 1, 1)
    frames = ((ramp.view(1, 1, 128, 1) + ramp + shift) % 256).to(torch.uint8)
    frames = frames.expand(8, 3, 128, 128).contiguous()
    video = SimpleNamespace(
        metadata=SimpleNamespace(average_fps=8.0, num_frames=8),
        get_frames_at=lambda indices: SimpleNamespace(data=frames[indices]),
    )
    # Every encode is over this limit, so the resolution is halved down to
    # the 64 px floor.
    monkeypatch.setattr(bedrock_models, "NOVA_MAX_INLINE_BYTES", 1)

    payload = nova._video_payload(video, "GENERIC_INDEX")
    raw = base64.b64decode(payload["singleEmbeddingParams"]["video"]["source"]["bytes"])
    metadata = VideoDecoder(raw).metadata
    assert metadata.codec == "h264"
    assert (metadata.height, metadata.width) == (32, 32)
