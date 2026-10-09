from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader

from mteb.abstasks.task_metadata import TaskMetadata
from mteb.mocks.mock_tasks import MockRetrievalTask
from mteb.models.model_implementations.videomae_models import VideoMAEWrapper

transformers = pytest.importorskip("transformers")

_NUM_FRAMES = 4
_SIZE = 32


class _StubDecoder:
    """VideoDecoder stand-in returning random uint8 NCHW frames."""

    def __init__(self, num_frames: int, seed: int) -> None:
        self.metadata = SimpleNamespace(num_frames=num_frames, end_stream_seconds=1.0)
        self._frames = torch.randint(
            0,
            256,
            (num_frames, 3, _SIZE, _SIZE),
            dtype=torch.uint8,
            generator=torch.Generator().manual_seed(seed),
        )

    def get_frames_at(self, indices: list[int]) -> SimpleNamespace:
        return SimpleNamespace(data=self._frames[indices])


def _save_tiny_checkpoint(path, model_cls) -> None:
    config = transformers.VideoMAEConfig(
        image_size=_SIZE,
        patch_size=16,
        num_frames=_NUM_FRAMES,
        tubelet_size=2,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        num_labels=3,
        use_mean_pooling=True,
    )
    torch.manual_seed(0)
    model = model_cls(config)
    with torch.no_grad():
        for param in model.parameters():
            # non-trivial weights, also for biases and the LayerNorm affine
            param.copy_(torch.randn_like(param) * 0.2 + (param == 1).float())
    model.save_pretrained(path)
    transformers.VideoMAEVideoProcessor(
        size={"shortest_edge": _SIZE}, crop_size={"height": _SIZE, "width": _SIZE}
    ).save_pretrained(path)


def _encode(wrapper: VideoMAEWrapper) -> torch.Tensor:
    rows = [{"video": _StubDecoder(6, seed)} for seed in range(2)]
    task_metadata: TaskMetadata = MockRetrievalTask.metadata
    out = wrapper.encode(
        DataLoader(rows, batch_size=2),
        task_metadata=task_metadata,
        hf_split="test",
        hf_subset="default",
        show_progress_bar=False,
    )
    return torch.from_numpy(out)


def test_finetuned_checkpoint_applies_fc_norm(tmp_path):
    """Fine-tuned VideoMAE features are fc_norm(mean(tokens)), the input of its classifier."""
    _save_tiny_checkpoint(tmp_path, transformers.VideoMAEForVideoClassification)
    wrapper = VideoMAEWrapper(str(tmp_path), revision=None, device="cpu")
    embeddings = _encode(wrapper)

    reference = transformers.VideoMAEForVideoClassification.from_pretrained(tmp_path)
    reference.eval()
    frames = [_StubDecoder(6, seed)._frames[[0, 2, 3, 5]] for seed in range(2)]
    pixel_values = wrapper.processor(frames, return_tensors="pt")["pixel_values"]
    with torch.inference_mode():
        logits = reference(pixel_values=pixel_values).logits
        assert torch.allclose(reference.classifier(embeddings), logits, atol=1e-4)


def test_pretraining_checkpoint_uses_plain_mean(tmp_path):
    _save_tiny_checkpoint(tmp_path, transformers.VideoMAEModel)
    wrapper = VideoMAEWrapper(str(tmp_path), revision=None, device="cpu")
    assert getattr(wrapper, "fc_norm", None) is None
    embeddings = _encode(wrapper)

    reference = transformers.VideoMAEModel.from_pretrained(tmp_path)
    reference.eval()
    frames = [_StubDecoder(6, seed)._frames[[0, 2, 3, 5]] for seed in range(2)]
    pixel_values = wrapper.processor(frames, return_tensors="pt")["pixel_values"]
    with torch.inference_mode():
        expected = reference(pixel_values=pixel_values).last_hidden_state.mean(dim=1)
    assert torch.allclose(embeddings, expected, atol=1e-4)
