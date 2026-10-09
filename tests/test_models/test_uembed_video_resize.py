from __future__ import annotations

import pytest
import torch

from mteb.models.model_implementations.uembed_models import _UEmbedInference


@pytest.mark.parametrize(
    ("frames", "height", "width", "expected_shape"),
    [
        # grids match UEmbed's reference (qwen-vl-utils, patch 16): hd 8 frames
        # [4,40,72], hd 64 frames [32,22,40], low-res 40 frames [20,20,28]
        (8, 1080, 1920, (8, 3, 640, 1152)),
        (64, 1080, 1920, (64, 3, 352, 640)),
        (40, 240, 320, (40, 3, 320, 448)),
        # odd frame counts are padded with the last frame, as the reference does
        (1, 240, 320, (2, 3, 320, 448)),
        (3, 360, 640, (4, 3, 352, 640)),
    ],
)
def test_uembed_video_resize_matches_reference(
    frames: int, height: int, width: int, expected_shape: tuple[int, ...]
) -> None:
    video = torch.randint(0, 256, (frames, 3, height, width), dtype=torch.uint8)
    resized = _UEmbedInference._resize_video(video)
    assert tuple(resized.shape) == expected_shape
    if frames % 2:
        assert torch.equal(resized[-1], resized[-2])
