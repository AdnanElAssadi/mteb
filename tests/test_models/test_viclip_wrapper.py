from __future__ import annotations

import torch

from mteb.models.model_implementations.viclip_models import ViCLIPWrapper


def test_frames_use_imagenet_normalisation() -> None:
    """ViCLIP's demo and training transforms normalise with ImageNet mean/std, not CLIP's."""
    frames = torch.full((2, 3, 224, 224), 128, dtype=torch.uint8)
    out = ViCLIPWrapper._preprocess_frames(frames)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    expected = (128 / 255.0 - mean) / std
    assert torch.allclose(out, expected.expand_as(out), atol=1e-5)
