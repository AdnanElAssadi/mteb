from __future__ import annotations

import base64
import io
import json
from types import SimpleNamespace

import numpy as np
import pytest

import mteb
from mteb.models.model_implementations import bedrock_models
from mteb.models.model_implementations.bedrock_models import (
    NovaMultimodalEmbeddingsModel,
)
from mteb.types import PromptType

pytest.importorskip("botocore", reason="boto3 is not installed")

NOVA_NAME = "bedrock/amazon-nova-2-multimodal-embeddings"
EMBEDDING_RESPONSE = {"embeddings": [{"embedding": [0.1, 0.2, 0.3]}]}


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


def test_nova_video_query_purpose_for_video_audio_documents() -> None:
    metadata = mteb.get_task("MSRVTTT2VA").metadata
    assert (
        NovaMultimodalEmbeddingsModel._purpose(metadata, PromptType.query)
        == "VIDEO_RETRIEVAL"
    )


def _recording_client(bodies: list[dict]) -> SimpleNamespace:
    def invoke_model(modelId: str, body: str) -> dict:  # noqa: N803
        bodies.append(json.loads(body))
        return {"body": io.BytesIO(json.dumps(EMBEDDING_RESPONSE).encode())}

    return SimpleNamespace(invoke_model=invoke_model)


def test_nova_video_audio_items_mux_audio_into_the_video(
    nova: NovaMultimodalEmbeddingsModel,
) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("av")
    try:
        from torchcodec.decoders import AudioDecoder, VideoDecoder
    except Exception:  # torchcodec missing or no FFmpeg libraries
        pytest.skip("torchcodec with FFmpeg is not available")

    ramp = torch.arange(64)
    frames = ((ramp.view(1, 1, 64, 1) + ramp) % 256).to(torch.uint8)
    frames = frames.expand(16, 3, 64, 64).contiguous()
    video = SimpleNamespace(
        metadata=SimpleNamespace(average_fps=8.0, num_frames=16),
        get_frames_at=lambda indices: SimpleNamespace(data=frames[indices]),
    )
    # 3 s, 440 Hz tone at 16 kHz; the 2 s video cuts it to 2 s
    tone = 0.5 * np.sin(2 * np.pi * 440 * np.arange(48_000) / 16_000)
    audio = {"array": tone.astype(np.float32), "sampling_rate": 16_000}

    bodies: list[dict] = []
    nova._client = _recording_client(bodies)
    embeddings = nova.encode(
        [{"video": [video, video], "audio": [audio, audio]}],
        task_metadata=mteb.get_task("AVEDatasetClassification").metadata,
        hf_split="test",
        hf_subset="default",
    )

    assert embeddings.shape == (2, 3)
    assert len(bodies) == 2  # one request per item
    params = bodies[0]["singleEmbeddingParams"]
    assert set(params) & {"text", "image", "audio", "video"} == {"video"}
    assert params["video"]["embeddingMode"] == "AUDIO_VIDEO_COMBINED"
    raw = base64.b64decode(params["video"]["source"]["bytes"])
    assert VideoDecoder(raw).metadata.num_frames == 16
    samples = AudioDecoder(raw).get_all_samples()
    assert samples.sample_rate == 16_000
    track = samples.data.numpy().mean(axis=0)
    assert abs(len(track) / 16_000 - 2.0) < 0.2
    # the AAC track carries the tone: 440 Hz spectral peak
    peak_hz = np.argmax(np.abs(np.fft.rfft(track))) * 16_000 / len(track)
    assert abs(peak_hz - 440) < 5


def test_nova_video_downscale_resizes_in_chunks(
    nova: NovaMultimodalEmbeddingsModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    try:
        from torchcodec.decoders import VideoDecoder
    except Exception:  # torchcodec missing or no FFmpeg libraries
        pytest.skip("torchcodec with FFmpeg is not available")

    ramp = torch.arange(128)
    frames = ((ramp.view(1, 1, 128, 1) + ramp) % 256).to(torch.uint8)
    frames = frames.expand(40, 3, 128, 128).contiguous()
    video = SimpleNamespace(
        metadata=SimpleNamespace(average_fps=8.0, num_frames=40),
        get_frames_at=lambda indices: SimpleNamespace(data=frames[indices]),
    )
    monkeypatch.setattr(bedrock_models, "NOVA_MAX_INLINE_BYTES", 1)
    interpolate = torch.nn.functional.interpolate
    resized_batch_sizes = []

    def recording_interpolate(x: torch.Tensor, **kwargs: object) -> torch.Tensor:
        resized_batch_sizes.append(x.shape[0])
        return interpolate(x, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "interpolate", recording_interpolate)

    payload = nova._video_payload(video, "GENERIC_INDEX")
    raw = base64.b64decode(payload["singleEmbeddingParams"]["video"]["source"]["bytes"])
    metadata = VideoDecoder(raw).metadata
    assert (metadata.num_frames, metadata.height, metadata.width) == (40, 32, 32)
    # a float copy of every frame at once is what blew up RAM on long 4K clips
    assert max(resized_batch_sizes) <= bedrock_models.NOVA_RESIZE_CHUNK_FRAMES
