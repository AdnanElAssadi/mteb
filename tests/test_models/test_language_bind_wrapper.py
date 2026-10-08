from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
from datasets import Dataset
from PIL import Image
from torch.utils.data import DataLoader

from mteb.mocks.mock_tasks import MockRetrievalTask
from mteb.models.model_implementations import language_bind_models as lb


class _FakeSubWrapper:
    """Records how the Omni wrapper builds its sub-models; text embedding = an id."""

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        self.model_name = model_name
        self.kwargs = kwargs

    def get_text_embeddings(self, inputs: DataLoader, **kwargs: Any) -> np.ndarray:
        tag = {"Video": 1.0, "Audio": 2.0, "Image": 3.0}
        value = next(v for k, v in tag.items() if k in self.model_name)
        return np.full((len(inputs.dataset), 2), value)


@pytest.fixture
def omni(monkeypatch: pytest.MonkeyPatch) -> lb.LanguageBindOmniWrapper:
    for name in [
        "LanguageBindVideoWrapper",
        "LanguageBindAudioWrapper",
        "LanguageBindImageWrapper",
    ]:
        monkeypatch.setattr(lb, name, _FakeSubWrapper)
    return lb.LanguageBindOmniWrapper("LanguageBind/LanguageBind_Omni", device="cpu")


def test_omni_pins_sub_model_revisions(omni: lb.LanguageBindOmniWrapper) -> None:
    for sub, meta in [
        (omni.video_model, lb.language_bind_video_ft),
        (omni.audio_model, lb.language_bind_audio_ft),
        (omni.image_model, lb.language_bind_image),
    ]:
        assert sub.model_name == meta.name
        assert sub.kwargs["revision"] == meta.revision


@pytest.mark.parametrize(
    ("modalities", "expected"),
    [
        # Audio_FT fine-tuned its own text tower
        (["text", "audio"], 2.0),
        # Video_FT and Image share one text tower
        (["text", "video"], 1.0),
        (["text", "image"], 1.0),
        (["text", "audio", "video"], 1.0),
    ],
)
def test_omni_text_tower_follows_task_modalities(
    omni: lb.LanguageBindOmniWrapper, modalities: list[str], expected: float
) -> None:
    task_metadata = MockRetrievalTask.metadata.model_copy(
        update={"modalities": modalities}
    )
    out = omni.encode(
        DataLoader(Dataset.from_dict({"text": ["a dog barks", "rain"]})),
        task_metadata=task_metadata,
        hf_split="test",
        hf_subset="default",
    )
    assert (out == expected).all()


class _FakeVisionModel(torch.nn.Module):
    def forward(self, pixel_values: torch.Tensor) -> tuple[None, torch.Tensor]:
        return None, pixel_values.mean(dim=(-2, -1))


@pytest.mark.parametrize("mode", ["RGB", "L", "RGBA", "P"])
def test_image_accepts_non_rgb_images(mode: str) -> None:
    """The LanguageBind transform normalises 3 channels; other modes used to crash."""
    from torchvision import transforms

    wrapper = lb.LanguageBindImageWrapper.__new__(lb.LanguageBindImageWrapper)
    wrapper.device = "cpu"
    wrapper.model = torch.nn.Module()
    wrapper.model.vision_model = _FakeVisionModel()
    wrapper.model.visual_projection = torch.nn.Identity()
    wrapper.processor = type(
        "P",
        (),
        {
            "transform": transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
        },
    )()
    image = Image.new("RGB", (8, 8), (200, 100, 50)).convert(mode)
    out = wrapper.get_image_embeddings(
        DataLoader(
            [{"image": image}],
            collate_fn=lambda rows: {"image": [r["image"] for r in rows]},
        ),
        show_progress_bar=False,
    )
    assert out.shape == (1, 3)
