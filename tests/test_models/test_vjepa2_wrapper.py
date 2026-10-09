"""V-JEPA2 classifier checkpoints embed with their trained attentive pooler.

Tiny random checkpoints are written to a temporary directory, so no weights are downloaded.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from torch.utils.data import DataLoader  # noqa: E402

from mteb.models.model_implementations import facebook_jepa  # noqa: E402

N_FRAMES = 4


def _save_checkpoint(path, classifier: bool) -> None:
    from transformers import (
        VJEPA2Config,
        VJEPA2ForVideoClassification,
        VJEPA2Model,
        VJEPA2VideoProcessor,
    )

    torch.manual_seed(0)
    config = VJEPA2Config(
        crop_size=32,
        frames_per_clip=N_FRAMES,
        hidden_size=32,
        num_attention_heads=2,
        num_hidden_layers=1,
        num_pooler_layers=1,
        pred_hidden_size=32,
        pred_num_attention_heads=2,
        pred_num_hidden_layers=1,
        num_labels=3,
    )
    model_cls = VJEPA2ForVideoClassification if classifier else VJEPA2Model
    model_cls(config).save_pretrained(path)
    VJEPA2VideoProcessor(crop_size=32).save_pretrained(path)


def _videos() -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(1)
    return [
        torch.randint(0, 256, (N_FRAMES, 3, 40, 48), generator=generator).to(
            torch.uint8
        )
        for _ in range(2)
    ]


def _encode(wrapper, videos, monkeypatch) -> torch.Tensor:
    # The collator decodes videos; here the frames are given directly.
    monkeypatch.setattr(
        facebook_jepa,
        "FramesCollator",
        lambda **_: lambda rows: {"video": [row["video"] for row in rows]},
    )
    loader = DataLoader([{"video": v} for v in videos], batch_size=2)
    return torch.from_numpy(
        wrapper.encode(
            loader,
            task_metadata=None,
            hf_split="test",
            hf_subset="default",
            show_progress_bar=False,
        )
    )


def _reference_inputs(path, videos):
    from transformers import AutoVideoProcessor

    return AutoVideoProcessor.from_pretrained(path)(videos=videos, return_tensors="pt")


def test_classifier_checkpoint_uses_trained_pooler(tmp_path, monkeypatch) -> None:
    from transformers import VJEPA2ForVideoClassification

    _save_checkpoint(tmp_path, classifier=True)
    videos = _videos()
    wrapper = facebook_jepa.VJepaV2Wrapper(str(tmp_path), revision=None, device="cpu")
    embeddings = _encode(wrapper, videos, monkeypatch)

    reference = VJEPA2ForVideoClassification.from_pretrained(tmp_path).eval()
    with torch.inference_mode():
        hidden = reference.vjepa2(
            **_reference_inputs(tmp_path, videos), skip_predictor=True
        ).last_hidden_state
        expected = reference.pooler(hidden)

    assert embeddings.shape == (2, 32)
    torch.testing.assert_close(embeddings, expected)
    # Mean pooling would ignore everything the classifier checkpoint adds.
    assert not torch.allclose(embeddings, hidden.mean(dim=1), atol=1e-3)


def test_pretrained_checkpoint_keeps_mean_pooling(tmp_path, monkeypatch) -> None:
    from transformers import VJEPA2Model

    _save_checkpoint(tmp_path, classifier=False)
    videos = _videos()
    wrapper = facebook_jepa.VJepaV2Wrapper(str(tmp_path), revision=None, device="cpu")
    assert getattr(wrapper, "pooler", None) is None
    embeddings = _encode(wrapper, videos, monkeypatch)

    reference = VJEPA2Model.from_pretrained(tmp_path).eval()
    with torch.inference_mode():
        expected = reference(
            **_reference_inputs(tmp_path, videos)
        ).last_hidden_state.mean(dim=1)
    torch.testing.assert_close(embeddings, expected)
