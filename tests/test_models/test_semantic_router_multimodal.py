"""multi-modal-embed-small text embeddings must not depend on batch padding.

Uses a tiny random BERT and a word-level tokenizer built locally, so nothing is
downloaded.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("tokenizers")

from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

from mteb.models.model_implementations.semantic_router_multimodal_models import (
    _multimodal_embed_small_class,
)

PAD, UNK = "[PAD]", "[UNK]"

TEXTS = [
    "a dog",
    "a cat sits on the mat while the dog runs in the park",
    "the park",
]


def _tiny_text_model() -> torch.nn.Module:
    words = sorted({w for t in TEXTS for w in t.split()})
    vocab = {PAD: 0, UNK: 1, **{w: i + 2 for i, w in enumerate(words)}}
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token=UNK))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token=PAD, unk_token=UNK
    )

    torch.manual_seed(0)
    config = BertConfig(
        vocab_size=len(vocab),
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=32,
    )
    encoder = BertModel(config).eval()

    base = _multimodal_embed_small_class()

    class _TinyTextOnly(base):
        def __init__(self) -> None:
            # skip the real __init__, which downloads the pretrained encoders
            super(base, self).__init__()
            self.text_tokenizer = tokenizer
            self.text_encoder = encoder

    return _TinyTextOnly()


@torch.no_grad()
def test_text_embeddings_do_not_depend_on_batch() -> None:
    model = _tiny_text_model()
    batched = model.encode_text(TEXTS)
    single = torch.cat([model.encode_text([t]) for t in TEXTS])
    torch.testing.assert_close(batched, single, atol=1e-5, rtol=1e-5)


@torch.no_grad()
def test_single_text_matches_model_card_mean() -> None:
    """For one unpadded text the card's plain mean over tokens is unchanged."""
    model = _tiny_text_model()
    for text in TEXTS:
        inputs = model.text_tokenizer([text], return_tensors="pt")
        hidden = model.text_encoder(**inputs).last_hidden_state
        expected = torch.nn.functional.normalize(hidden.mean(dim=1), p=2, dim=-1)
        torch.testing.assert_close(model.encode_text([text]), expected)
