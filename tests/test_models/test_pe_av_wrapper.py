from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from mteb.mocks.mock_tasks import MockRetrievalTask
from mteb.models.model_implementations.pe_av_models import PEAudioVisualWrapper

transformers = pytest.importorskip("transformers")

_HIDDEN = 16
_DIM = 32
_SR = 16_000


def _linear(n_in: int, n_out: int, seed: int) -> nn.Linear:
    torch.manual_seed(seed)
    return nn.Linear(n_in, n_out)


class _FakeTextModel(nn.Module):
    """ModernBERT stand-in: on transformers >= 5.1 hidden_states[-1] is post final_norm."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.embed = nn.Embedding(100, _HIDDEN)
        self.final_norm = nn.LayerNorm(_HIDDEN)
        with torch.no_grad():
            self.final_norm.weight.uniform_(0.5, 2.0)
            self.final_norm.bias.normal_()

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs: Any
    ) -> SimpleNamespace:
        hidden = self.embed(input_ids)
        last = self.final_norm(hidden)
        return SimpleNamespace(last_hidden_state=last, hidden_states=(hidden, last))


class PeAudioDacEncoder(nn.Module):  # same class name as in transformers
    def __init__(self) -> None:
        super().__init__()
        self.seen_dtypes: list[torch.dtype] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.seen_dtypes.append(x.dtype)
        return x


class _FakeAudioEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dac_encoder = PeAudioDacEncoder()
        self.proj = _linear(1, _HIDDEN, 10)

    def forward(
        self, input_values: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> SimpleNamespace:
        pooled = self.proj(input_values.float().mean(dim=-1, keepdim=True))
        return SimpleNamespace(pooler_output=pooled)


class _FakePeAudioVideoModel(nn.Module):
    calls: list[dict[str, Any]] = []

    def __init__(self) -> None:
        super().__init__()
        self.text_model = _FakeTextModel()
        self.text_audio_video_head = _linear(_HIDDEN, _DIM, 1)
        self.video_model = nn.Module()
        self.video_model.text_video_head = _linear(_HIDDEN, _DIM, 2)
        self.audio_model = nn.Module()
        self.audio_model.text_audio_head = _linear(_HIDDEN, _DIM, 3)
        self.audio_model.audio_encoder = _FakeAudioEncoder()
        self.audio_model.audio_head = _linear(_HIDDEN, _DIM, 4)
        self.audio_plus_text_head = _linear(2 * _HIDDEN, _DIM, 5)
        self.video_plus_text_head = _linear(2 * _HIDDEN, _DIM, 6)

    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> _FakePeAudioVideoModel:
        cls.calls.append(kwargs)
        return cls()

    def get_audio_embeds(
        self, input_values: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        pooled = self.audio_model.audio_encoder(input_values, padding_mask)
        return self.audio_model.audio_head(pooled.pooler_output)


class _FakeProcessor:
    calls: list[dict[str, Any]] = []

    def __init__(self) -> None:
        self.feature_extractor = SimpleNamespace(sampling_rate=_SR)
        self.text_kwargs: dict[str, Any] = {}

    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> _FakeProcessor:
        cls.calls.append(kwargs)
        return cls()

    def __call__(
        self,
        text: list[str] | None = None,
        audio: list[np.ndarray] | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        if text is not None:
            self.text_kwargs = kwargs
            return _tokenize(text)
        assert audio is not None
        length = max(len(a) for a in audio)
        values = torch.zeros(len(audio), length)
        for i, a in enumerate(audio):
            values[i, : len(a)] = torch.as_tensor(a)
        return {"input_values": values}


def _tokenize(texts: list[str]) -> dict[str, torch.Tensor]:
    length = max(len(t) for t in texts)
    ids = torch.zeros(len(texts), length, dtype=torch.long)
    for i, t in enumerate(texts):
        ids[i, : len(t)] = torch.tensor([ord(c) % 100 for c in t])
    return {"input_ids": ids, "attention_mask": (ids != 0).long()}


class _Rows(list):
    """Minimal map-style dataset exposing ``features`` like a datasets.Dataset."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        super().__init__(rows)
        self.features = dict.fromkeys(rows[0])


_TEXTS = ["a dog barks", "rain on a roof", "someone plays the piano"]


@pytest.fixture
def wrapper(monkeypatch: pytest.MonkeyPatch) -> PEAudioVisualWrapper:
    for name, fake in [
        ("PeAudioVideoModel", _FakePeAudioVideoModel),
        ("PeAudioVideoProcessor", _FakeProcessor),
    ]:
        real = getattr(transformers, name)
        monkeypatch.setattr(real, "from_pretrained", staticmethod(fake.from_pretrained))
    _FakePeAudioVideoModel.calls.clear()
    _FakeProcessor.calls.clear()
    return PEAudioVisualWrapper("facebook/pe-av-small", revision="abc", device="cpu")


def _encode(wrapper: PEAudioVisualWrapper, rows: list[dict], category: str):
    task_metadata = MockRetrievalTask.metadata.model_copy(update={"category": category})
    return wrapper.encode(
        DataLoader(_Rows(rows), batch_size=2),
        task_metadata=task_metadata,
        hf_split="test",
        hf_subset="default",
        show_progress_bar=False,
    )


def _text_cls_pre_norm(model: _FakePeAudioVideoModel) -> torch.Tensor:
    return model.text_model.embed(_tokenize(_TEXTS)["input_ids"])[:, 0]


def _normalize(x: torch.Tensor) -> np.ndarray:
    return torch.nn.functional.normalize(x.float(), dim=-1).detach().numpy()


def _assert_close(a: np.ndarray, b: np.ndarray) -> None:
    cos = (a * b).sum(-1) / np.linalg.norm(a, axis=-1) / np.linalg.norm(b, axis=-1)
    assert cos.min() > 0.999, cos


@pytest.mark.parametrize(
    ("category", "head"),
    [
        ("t2v", "video_model.text_video_head"),
        ("v2t", "video_model.text_video_head"),
        ("t2a", "audio_model.text_audio_head"),
        ("a2t", "audio_model.text_audio_head"),
        ("t2va", "text_audio_video_head"),
        ("va2t", "text_audio_video_head"),
    ],
)
def test_text_head_and_pooling_follow_meta(
    wrapper: PEAudioVisualWrapper, category: str, head: str
) -> None:
    """Text uses the CLS token before ModernBERT's final norm and the head for the task's other modality."""
    model = wrapper.model
    out = _encode(wrapper, [{"text": t} for t in _TEXTS], category)
    expected = _normalize(model.get_submodule(head)(_text_cls_pre_norm(model)))
    _assert_close(out, expected)
    assert wrapper.processor.text_kwargs.get("max_length") == 512


def test_audio_plus_text_query_uses_fused_head(wrapper: PEAudioVisualWrapper) -> None:
    model = wrapper.model
    rng = np.random.default_rng(0)
    audio = [rng.standard_normal(_SR // 2).astype(np.float32) for _ in range(3)]
    rows = [
        {"text": t, "audio": {"array": a, "sampling_rate": _SR}}
        for t, a in zip(_TEXTS, audio, strict=True)
    ]
    out = _encode(wrapper, rows, "at2v")

    with torch.no_grad():
        media = torch.cat(
            [
                model.audio_model.audio_encoder(torch.as_tensor(a)[None]).pooler_output
                for a in audio
            ]
        )
        fused = model.audio_plus_text_head(
            torch.cat([media, _text_cls_pre_norm(model)], dim=-1)
        )
    _assert_close(out, _normalize(fused))


def test_init_passes_revision_and_runs_dac_in_fp32(
    wrapper: PEAudioVisualWrapper,
) -> None:
    assert _FakePeAudioVideoModel.calls[-1].get("revision") == "abc"
    assert _FakeProcessor.calls[-1].get("revision") == "abc"

    dac = wrapper.model.audio_model.audio_encoder.dac_encoder
    with torch.autocast("cpu", dtype=torch.bfloat16):
        dac(torch.ones(1, 4, dtype=torch.bfloat16))
    assert dac.seen_dtypes == [torch.float32]
