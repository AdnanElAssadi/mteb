"""OmniRetriever embeds text-only inputs like the authors' ``text_embeds`` (no weights needed)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from transformers import BatchEncoding  # noqa: E402

from mteb.models.model_implementations.omniretriever_models import (  # noqa: E402
    OmniRetrieverWrapper,
)

HIDDEN = 8
IM_END = 99


class _Tokenizer:
    """Whitespace tokenizer; ``<|im_end|>`` gets its own id. Pads on the left with 0."""

    def __call__(self, texts, padding, padding_side, return_tensors) -> BatchEncoding:
        assert padding_side == "left"
        rows = []
        for text in texts:
            body, end, _ = text.partition("<|im_end|>")
            rows.append([1 + len(w) for w in body.split()] + ([IM_END] if end else []))
        width = max(map(len, rows))
        ids = [[0] * (width - len(r)) + r for r in rows]
        mask = [[0] * (width - len(r)) + [1] * len(r) for r in rows]
        return BatchEncoding(
            {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask)}
        )


class _Decoder(torch.nn.Module):
    """Decoder stand-in whose last-token output depends on the token ids and positions."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = []

    def forward(self, inputs_embeds, attention_mask, position_ids=None, **kwargs):  # noqa: ANN003
        self.calls.append(
            {"attention_mask": attention_mask, "position_ids": position_ids}
        )
        if position_ids is None:  # the decoder's default: 0..L-1 regardless of padding
            position_ids = torch.arange(inputs_embeds.shape[1]).expand(
                inputs_embeds.shape[0], -1
            )
        weights = (attention_mask * (position_ids + 1)).unsqueeze(-1)
        pooled = (inputs_embeds * weights).cumsum(dim=1)
        return (pooled,)


class _Backbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.embed = torch.nn.Embedding(128, HIDDEN)
        self.model = _Decoder()
        self.fused_forward_calls = 0

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.embed

    def forward(self, input_ids, pred_embeds=False, **kwargs):  # noqa: ANN003
        # full forward + classify_linear fusion head (mllm_embeds)
        self.fused_forward_calls += 1
        return SimpleNamespace(mllm_embeds=torch.ones(input_ids.shape[0], HIDDEN))


class _Peft(torch.nn.Module):
    def __init__(self, backbone: _Backbone) -> None:
        super().__init__()
        self.backbone = backbone

    def get_base_model(self) -> _Backbone:
        return self.backbone

    def forward(self, **kwargs):  # noqa: ANN003
        return self.backbone(**kwargs)


def _wrapper() -> OmniRetrieverWrapper:
    wrapper = OmniRetrieverWrapper.__new__(OmniRetrieverWrapper)
    wrapper.device = "cpu"
    wrapper.tokenizer = _Tokenizer()
    wrapper.model = _Peft(_Backbone())
    wrapper.processor = SimpleNamespace(
        apply_chat_template=lambda messages, **_: [
            "<|im_start|>user\n" + messages[0]["content"][-1]["text"] + "<|im_end|>\n"
        ],
        replace_multimodal_special_tokens=lambda prompts, *args, **kwargs: prompts,
    )
    return wrapper


def _authors_text_embedding(backbone: _Backbone, caption: str) -> torch.Tensor:
    """Authors' label_ids branch, one caption at a time (their eval uses batch_size=1)."""
    ids = _Tokenizer()([caption + "<|im_end|>"], True, "left", "pt")["input_ids"]
    hidden = backbone.model(
        inputs_embeds=backbone.get_input_embeddings()(ids),
        attention_mask=torch.ones_like(ids),
    )[0]
    return torch.nn.functional.normalize(hidden[:, -1, :], dim=-1)


def test_text_only_inputs_use_the_text_embeds_path() -> None:
    wrapper = _wrapper()
    captions = ["a dog barks twice", "rain", "a man speaks while birds chirp loudly"]
    with torch.inference_mode():
        out = wrapper._encode_batch({"text": captions})
        expected = torch.cat(
            [_authors_text_embedding(wrapper.model.backbone, c) for c in captions]
        )
    assert wrapper.model.backbone.fused_forward_calls == 0
    torch.testing.assert_close(out, expected)


def test_text_embedding_does_not_depend_on_padding() -> None:
    wrapper = _wrapper()
    with torch.inference_mode():
        alone = wrapper._encode_batch({"text": ["rain"]})
        batched = wrapper._encode_batch(
            {"text": ["rain", "a man speaks while birds chirp loudly"]}
        )
    torch.testing.assert_close(batched[:1], alone)
