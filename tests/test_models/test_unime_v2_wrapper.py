"""UniME-V2 resizes images to 1344x1344 like the official evaluation (no weights needed)."""

from __future__ import annotations

import pytest

Image = pytest.importorskip("PIL.Image")

from mteb.models.model_implementations.unime_v2_models import (  # noqa: E402
    UniMEV2Wrapper,
)


@pytest.mark.parametrize("size", [(640, 480), (300, 200), (2000, 1500)])
def test_images_are_resized_like_the_official_eval(size: tuple[int, int]) -> None:
    image = Image.new("RGB", size, color=(10, 200, 30))
    conversation = UniMEV2Wrapper._build_conversation(
        text=None, image=image, video=None, instruction=""
    )
    (image_part,) = [
        part for part in conversation[0]["content"] if part["type"] == "image"
    ]
    assert image_part["image"].size == (1344, 1344)
    # official process_image: a plain PIL resize to 1344x1344
    assert list(image_part["image"].getdata()) == list(
        image.resize((1344, 1344)).getdata()
    )
