from __future__ import annotations

import math
import unicodedata
from typing import TYPE_CHECKING, Any

from typing_extensions import override

from mteb.models.instruct_wrapper import InstructSentenceTransformerModel
from mteb.models.model_meta import ModelMeta, ScoringFunction

if TYPE_CHECKING:
    import torch
    from transformers import Qwen3VLVideoProcessor

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import PromptType

QWEN3_VL_EMBEDDING_CITATION = """@article{qwen3vlembedding,
  title={Qwen3-VL-Embedding and Qwen3-VL-Reranker: A Unified Framework for State-of-the-Art Multimodal Retrieval and Ranking},
  author={Li, Mingxin and Zhang, Yanzhao and Long, Dingkun and Chen Keqin and Song, Sibo and Bai, Shuai and Yang, Zhibo and Xie, Pengjun and Yang, An and Liu, Dayiheng and Zhou, Jingren and Lin, Junyang},
  journal={arXiv preprint arXiv:2601.04720},
  year={2026}
}"""

IMAGE_BASE_FACTOR = 16
IMAGE_FACTOR = IMAGE_BASE_FACTOR * 2
MIN_PIXELS = 4 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_PIXELS = 1800 * IMAGE_FACTOR * IMAGE_FACTOR
DEFAULT_INSTRUCTION = "Represent the user's input."
# Video budget of the official scripts, applied by qwen-vl-utils' fetch_video to
# frame lists: 128-768 tokens per frame, and total_pixels over the whole clip
# https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B/blob/main/scripts/qwen3_vl_embedding.py
VIDEO_FRAME_MIN_PIXELS = 128 * IMAGE_FACTOR * IMAGE_FACTOR
VIDEO_FRAME_MAX_PIXELS = 768 * IMAGE_FACTOR * IMAGE_FACTOR
VIDEO_MAX_TOTAL_PIXELS = 10 * VIDEO_FRAME_MAX_PIXELS


def _qwen_vl_utils_frame_size(
    num_frames: int, height: int, width: int, factor: int, total_pixels: int
) -> tuple[int, int]:
    """Per-frame size qwen-vl-utils' fetch_video resizes a frame list to."""
    num_frames += num_frames % 2  # frame lists are padded to an even count
    max_pixels = max(
        min(VIDEO_FRAME_MAX_PIXELS, total_pixels / num_frames * 2),
        int(VIDEO_FRAME_MIN_PIXELS * 1.05),
    )
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt(height * width / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < VIDEO_FRAME_MIN_PIXELS:
        beta = math.sqrt(VIDEO_FRAME_MIN_PIXELS / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def load_frame_budget_video_processor(
    model_name: str, *, total_pixels: int, **kwargs: Any
) -> Qwen3VLVideoProcessor:
    """Load the Qwen3-VL video processor, resizing frames as the official scripts do.

    The processor's own `size` is a budget for the whole clip, so few-frame
    clips keep near-native resolution and small frames are not upscaled. The
    official scripts resize every frame with qwen-vl-utils instead (and call
    the processor with do_resize=False).
    """
    from torchvision.transforms.functional import InterpolationMode, resize
    from transformers import Qwen3VLVideoProcessor

    class _FrameBudgetVideoProcessor(Qwen3VLVideoProcessor):
        @override
        def resize(  # type: ignore[override]
            self, videos: torch.Tensor, factor: int, **resize_kwargs: Any
        ) -> torch.Tensor:
            batch, num_frames, _, height, width = videos.shape
            frame_size = _qwen_vl_utils_frame_size(
                num_frames, height, width, factor, total_pixels
            )
            frames = resize(
                videos.flatten(0, 1),
                list(frame_size),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            )
            return frames.view(batch, num_frames, *frames.shape[1:])

    return _FrameBudgetVideoProcessor.from_pretrained(model_name, **kwargs)


class Qwen3VLEmbeddingWrapper(InstructSentenceTransformerModel):
    # Qwen3-VL renders frame timestamps into the prompt; without the real source
    # fps and frame indices the processor assumes 24 fps
    video_metadata = True

    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        device: str | None = None,
        min_pixels: int = MIN_PIXELS,
        max_pixels: int = MAX_PIXELS,
        # fps=1, 64 frames: FPS=1 and MAX_FRAMES=64 in the official script (the
        # generic Qwen3-VL video processor default is fps=2)
        # https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B/blob/main/scripts/qwen3_vl_embedding.py
        fps: float | None = 1.0,
        max_frames: int | None = 64,
        num_frames: int | None = None,
        **kwargs: Any,
    ) -> None:
        from transformers import AutoImageProcessor

        processor_kwargs = kwargs.get("processor_kwargs", {})

        image_processor = AutoImageProcessor.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=True,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            **kwargs,
        )
        # VideoCollator already samples frames; skip inner sampling.
        video_processor = load_frame_budget_video_processor(
            model_name,
            total_pixels=VIDEO_MAX_TOTAL_PIXELS,
            revision=revision,
            trust_remote_code=True,
            do_sample_frames=False,
            **kwargs,
        )
        processor_kwargs |= {
            "video_processor": video_processor,
            "image_processor": image_processor,
        }

        kwargs["processor_kwargs"] = processor_kwargs

        super().__init__(
            model_name,
            revision=revision,
            device=device,
            fps=fps,
            max_frames=max_frames,
            num_frames=num_frames,
            **kwargs,
        )

    def get_task_instruction(
        self,
        task_metadata: TaskMetadata,
        prompt_type: PromptType | None,
    ) -> str:
        instruction = super().get_task_instruction(task_metadata, prompt_type)
        instruction = instruction or DEFAULT_INSTRUCTION
        instruction = instruction.strip()
        # Checks if the last character is not punctuation and appends "." then
        if instruction and not unicodedata.category(instruction[-1]).startswith("P"):
            instruction += "."
        return instruction


qwen3_vl_embedding_2b = ModelMeta(
    loader=Qwen3VLEmbeddingWrapper,
    name="Qwen/Qwen3-VL-Embedding-2B",
    model_type=["dense"],
    languages=["eng-Latn"],
    open_weights=True,
    revision="9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda",
    release_date="2026-01-08",
    modalities=["image", "text", "video"],
    n_parameters=2_127_532_032,
    n_embedding_parameters=311_164_928,
    memory_usage_mb=7629,
    embed_dim=2048,
    license="apache-2.0",
    max_tokens=32768,
    reference="https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch", "Transformers", "safetensors"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    citation=QWEN3_VL_EMBEDDING_CITATION,
    extra_requirements_groups=["multimodal-sbert"],
)

qwen3_vl_embedding_8b = ModelMeta(
    loader=Qwen3VLEmbeddingWrapper,
    name="Qwen/Qwen3-VL-Embedding-8B",
    model_type=["dense"],
    languages=["eng-Latn"],
    open_weights=True,
    revision="2c4565515e0f265c6511776e7193b22c0968ddc7",
    release_date="2026-01-08",
    modalities=["image", "text", "video"],
    n_parameters=8_144_793_840,
    n_embedding_parameters=622_329_856,
    memory_usage_mb=30518,
    embed_dim=4096,
    license="apache-2.0",
    max_tokens=32768,
    reference="https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch", "Transformers", "safetensors"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    citation=QWEN3_VL_EMBEDDING_CITATION,
    extra_requirements_groups=["multimodal-sbert"],
)


dna_vl_steer_2b = ModelMeta(
    loader=Qwen3VLEmbeddingWrapper,
    name="dnotitia/DNA-VL-STEER-2B",
    adapted_from="Qwen/Qwen3-VL-Embedding-2B",
    model_type=["dense"],
    # Calibrated on all 36 XM3600 languages (the base entry declares eng-Latn only).
    languages=[
        "eng-Latn",
        "fra-Latn",
        "deu-Latn",
        "spa-Latn",
        "zho-Hans",
        "kor-Hang",
        "ben-Beng",
        "fil-Latn",
        "hin-Deva",
        "swa-Latn",
        "tel-Telu",
        "jpn-Jpan",
        "rus-Cyrl",
        "ara-Arab",
        "vie-Latn",
        "tur-Latn",
        "tha-Thai",
        "ell-Grek",
        "fas-Arab",
        "fin-Latn",
        "ind-Latn",
        "heb-Hebr",
        "pol-Latn",
        "ukr-Cyrl",
        "nld-Latn",
        "ita-Latn",
        "por-Latn",
        "ces-Latn",
        "swe-Latn",
        "hun-Latn",
        "ron-Latn",
        "hrv-Latn",
        "dan-Latn",
        "nor-Latn",
        "mri-Latn",
        "quz-Latn",
    ],
    open_weights=True,
    revision="521a9cead4b31920579f80dfc9609a7969ddba86",
    release_date="2026-08-27",
    modalities=["image", "text", "video"],
    n_parameters=2_127_532_032,
    n_embedding_parameters=311_164_928,
    memory_usage_mb=7629,
    embed_dim=2048,
    license="apache-2.0",
    max_tokens=32768,
    reference="https://huggingface.co/dnotitia/DNA-VL-STEER-2B",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch", "Transformers", "safetensors"],
    use_instructions=True,
    public_training_code=None,
    # Calibration corpus is machine-translated CC12M captions (36 languages, ~9.5M rows).
    # It contains no MTEB task data, so there is no leaderboard contamination.
    public_training_data=None,
    training_datasets=None,
    extra_requirements_groups=["multimodal-sbert"],
)
