from __future__ import annotations

from typing import Any

import pytest
import torch

from mteb.models.modality_collators import VideoFrameMetadata

METADATA = VideoFrameMetadata(
    total_num_frames=1350,
    fps=30.0,
    duration=45.0,
    frames_indices=[0, 450, 900, 1349],
)


def test_uembed_passes_collator_metadata() -> None:
    pytest.importorskip("transformers.video_utils")
    from mteb.models.model_implementations.uembed_models import _UEmbedInference

    seen: dict[str, Any] = {}

    class _Processor:
        def apply_chat_template(
            self, conversations: list[Any], **kwargs: Any
        ) -> list[str]:
            return ["prompt"] * len(conversations)

        def __call__(self, **kwargs: Any) -> dict[str, Any]:
            seen.update(kwargs)
            return {}

    inference = object.__new__(_UEmbedInference)
    inference.processor = _Processor()
    inference.fps = 1.0
    inference.max_length = 128
    inference.device = "cpu"
    video = torch.zeros(4, 3, 8, 8, dtype=torch.uint8)
    inference._preprocess_inputs([[]], [], [video], [METADATA])
    (metadata,) = seen["video_metadata"]
    assert metadata.frames_indices == [0, 450, 900, 1349]
    assert metadata.timestamps[-1] == pytest.approx(1349 / 30)


def test_wemm_video_message_carries_frame_rate() -> None:
    from mteb.models.model_implementations.wemm_models import _video_message

    message = _video_message({"video": ["f0", "f1", "f2", "f3"], "text": "q"}, METADATA)
    assert message == [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": ["f0", "f1", "f2", "f3"],
                    "sample_fps": pytest.approx(4 * 30 / 1350),
                },
                {"type": "text", "text": "q"},
            ],
        }
    ]


def test_wemm_video_message_keeps_nominal_rate_on_normal_clips() -> None:
    from mteb.models.model_implementations.wemm_models import _video_message

    # a 10 s, 30 fps clip sampled at 2 fps: qwen-vl-utils reports exactly 2.0
    metadata = VideoFrameMetadata(
        total_num_frames=300,
        fps=30.0,
        duration=10.0,
        frames_indices=[round(i * 299 / 19) for i in range(20)],
    )
    (message,) = _video_message({"video": ["f"] * 20}, metadata)
    assert message["content"][0]["sample_fps"] == pytest.approx(2.0)
