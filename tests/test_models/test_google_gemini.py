from __future__ import annotations

import io
import wave
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest

from mteb.models.model_implementations import google_gemini
from mteb.models.model_implementations.google_gemini import (
    GoogleGeminiEmbeddingModel,
    _audio_to_wav_bytes,
    _format_gemini_embedding_2_text,
)
from mteb.types import PromptType

try:
    import httpx
    from google.genai import errors as genai_errors
except ImportError:
    genai_errors = None

requires_genai = pytest.mark.skipif(
    genai_errors is None, reason="google-genai is not installed"
)


@pytest.mark.parametrize(
    ("google_task_type", "expected"),
    [
        ("RETRIEVAL_QUERY", "task: search result | query: example text"),
        ("QUESTION_ANSWERING", "task: question answering | query: example text"),
        ("FACT_VERIFICATION", "task: fact checking | query: example text"),
        ("CLASSIFICATION", "task: classification | query: example text"),
        ("CLUSTERING", "task: clustering | query: example text"),
        ("SEMANTIC_SIMILARITY", "task: sentence similarity | query: example text"),
    ],
)
def test_gemini_embedding_2_formats_task_prefixes(
    google_task_type: str, expected: str
) -> None:
    assert (
        _format_gemini_embedding_2_text(
            "example text", google_task_type, PromptType.query
        )
        == expected
    )


def test_gemini_embedding_2_formats_documents_with_title() -> None:
    assert (
        _format_gemini_embedding_2_text(
            "example text", "FACT_VERIFICATION", PromptType.document, "Example"
        )
        == "title: Example | text: example text"
    )


def test_gemini_embedding_2_formats_documents_without_title() -> None:
    assert (
        _format_gemini_embedding_2_text(
            "example text", "FACT_VERIFICATION", PromptType.document
        )
        == "title: none | text: example text"
    )


def test_gemini_embedding_2_leaves_unknown_task_type_unchanged() -> None:
    assert _format_gemini_embedding_2_text("example text", None, None) == "example text"


def test_gemini_audio_clips_instead_of_wrapping() -> None:
    array = np.array([1.5, 1.001, -1.02, 0.5], dtype=np.float32)
    wav_bytes = _audio_to_wav_bytes({"array": array, "sampling_rate": 16000})
    with wave.open(io.BytesIO(wav_bytes)) as wf:
        pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    np.testing.assert_array_equal(pcm, [32767, 32767, -32767, 16383])


def _fake_response(values: list[float]) -> SimpleNamespace:
    return SimpleNamespace(embeddings=[SimpleNamespace(values=values)])


def _mocked_gemini_model(side_effect: list) -> tuple[GoogleGeminiEmbeddingModel, Mock]:
    """Gemini wrapper whose async client returns or raises ``side_effect`` in order."""
    embed_content = AsyncMock(side_effect=side_effect)
    model = object.__new__(GoogleGeminiEmbeddingModel)
    model.model_name = "gemini-embedding-2"
    model.embed_dim = 3
    model.client = Mock()
    model.client.aio.models.embed_content = embed_content
    return model, embed_content


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    recorded: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        recorded.append(seconds)

    monkeypatch.setattr(google_gemini.asyncio, "sleep", fake_sleep)
    return recorded


@requires_genai
@pytest.mark.parametrize(
    "error",
    [
        pytest.param(lambda: genai_errors.ClientError(400, {}), id="400"),
        pytest.param(lambda: genai_errors.ClientError(403, {}), id="403"),
        pytest.param(lambda: ValueError("bad local input"), id="local"),
    ],
)
def test_gemini_fails_fast_on_non_retryable_errors(error, sleeps) -> None:
    model, embed_content = _mocked_gemini_model([error(), _fake_response([1.0])])
    with pytest.raises(type(error())):
        model._embed(["text"])
    assert embed_content.await_count == 1
    assert sleeps == []


@requires_genai
@pytest.mark.parametrize(
    "error",
    [
        pytest.param(lambda: genai_errors.ServerError(503, {}), id="503"),
        pytest.param(lambda: genai_errors.ClientError(429, {}), id="429"),
        pytest.param(lambda: httpx.ReadTimeout("timed out"), id="timeout"),
    ],
)
def test_gemini_retries_transient_errors_with_backoff(error, sleeps) -> None:
    model, embed_content = _mocked_gemini_model(
        [error(), error(), error(), _fake_response([1.0, 2.0, 3.0])]
    )
    np.testing.assert_array_equal(model._embed(["text"]), [[1.0, 2.0, 3.0]])
    assert embed_content.await_count == 4
    assert len(sleeps) == 3
    # exponential backoff (1, 2, 4 s) plus < 1 s jitter
    for slept, base in zip(sleeps, [1, 2, 4], strict=True):
        assert base <= slept < base + 1


@requires_genai
def test_gemini_rate_limit_uses_server_retry_hint(sleeps) -> None:
    error = genai_errors.ClientError(
        429, {"error": {"message": "Quota exceeded. Please retry in 7.5s."}}
    )
    model, _ = _mocked_gemini_model([error, _fake_response([1.0])])
    model._embed(["text"])
    assert len(sleeps) == 1
    assert 7.5 <= sleeps[0] < 8.5


@requires_genai
def test_gemini_gives_up_after_max_attempts(sleeps) -> None:
    errors = [genai_errors.ServerError(500, {}) for _ in range(10)]
    model, embed_content = _mocked_gemini_model(errors)
    with pytest.raises(genai_errors.ServerError):
        model._embed(["text"])
    assert embed_content.await_count == 10
    assert len(sleeps) == 10 - 1
    assert max(sleeps) < 61
