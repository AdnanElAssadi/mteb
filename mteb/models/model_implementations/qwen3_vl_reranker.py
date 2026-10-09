from __future__ import annotations

import unicodedata
from typing import TYPE_CHECKING, Any

from mteb.models.model_implementations.qwen3_vl_embedding_models import (
    IMAGE_FACTOR,
    QWEN3_VL_EMBEDDING_CITATION,
    load_frame_budget_video_processor,
)
from mteb.models.model_meta import ModelMeta
from mteb.models.sentence_transformer_wrapper import CrossEncoderWrapper

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType

DEFAULT_INSTRUCTION = "Retrieve images or text relevant to the user's query."
# MAX_TOTAL_PIXELS of the official script: 4 frames of 1280 tokens, times FRAME_FACTOR
# https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B/blob/main/scripts/qwen3_vl_reranker.py
VIDEO_MAX_TOTAL_PIXELS = 4 * 2 * 1280 * IMAGE_FACTOR * IMAGE_FACTOR


def _snapshot_with_module_paths(
    model: str, revision: str | None, module_paths: dict[str, str]
) -> str:
    """Return a local view of the checkpoint whose modules.json uses corrected module paths.

    Every file is symlinked from the Hugging Face cache snapshot; only modules.json is rewritten.
    """
    import json
    import tempfile
    from pathlib import Path

    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(model, revision=revision))
    modules = json.loads((snapshot / "modules.json").read_text())
    for module in modules:
        module["path"] = module_paths.get(module["path"], module["path"])
    local = Path(tempfile.mkdtemp(prefix=f"mteb_{model.replace('/', '__')}_"))
    for entry in snapshot.iterdir():
        if entry.name != "modules.json":
            (local / entry.name).symlink_to(entry)
    (local / "modules.json").write_text(json.dumps(modules, indent=2))
    return str(local)


QWEN3_VL_LANGUAGES = [
    "eng-Latn",
    "cmn-Hans",
    "spa-Latn",
    "fra-Latn",
    "ara-Arab",
    "por-Latn",
    "rus-Cyrl",
    "urd-Arab",
    "ind-Latn",
    "deu-Latn",
    "jpn-Jpan",
    "vie-Latn",
    "tur-Latn",
    "kor-Hang",
    "fas-Arab",
    "ita-Latn",
    "tha-Thai",
    "pol-Latn",
    "ukr-Cyrl",
    "uzb-Latn",
    "ron-Latn",
    "nld-Latn",
    "kaz-Cyrl",
    "ell-Grek",
    "ces-Latn",
    "swe-Latn",
    "srp-Cyrl",
    "heb-Hebr",
    "dan-Latn",
    "fin-Latn",
    "nor-Latn",
    "slv-Latn",
    "gle-Latn",
]


class Qwen3VLRerankerWrapper(CrossEncoderWrapper):
    """Wrapper for the Qwen3-VL-Reranker series.

    The reranker scores a (query, document) pair jointly, where either side may
    be text, an image, a video, or a mixture of these. It is exposed as a
    standard `sentence_transformers.CrossEncoder`, so the multimodal input
    collection in `CrossEncoderWrapper` is reused as-is; this subclass only
    supplies the task-level instruction. Pass `use_instructions=False` to
    score pairs without one.

    Reference implementation: https://github.com/QwenLM/Qwen3-VL-Embedding
    """

    def __init__(  # noqa: PLR0913, PLR0917
        self,
        model: str,
        revision: str | None = None,
        device: str | None = None,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        # fps=1, 64 frames: FPS=1 and MAX_FRAMES=64 in the official script (the
        # generic Qwen3-VL video processor default is fps=2)
        # https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B/blob/main/scripts/qwen3_vl_reranker.py
        fps: float | None = 1.0,
        max_frames: int | None = 64,
        num_frames: int | None = None,
        use_instructions: bool = True,
        # resize video frames as the official script does (qwen-vl-utils
        # per-frame budget) instead of the processor's whole-clip budget
        video_frame_budget: bool = False,
        *,
        module_paths: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        if module_paths:
            model = _snapshot_with_module_paths(model, revision, module_paths)
            revision = None
        # Default to the checkpoint's own preprocessor_config.json. Forcing a
        # pixel budget changes how images are resized and measurably shifts
        # scores away from the reference implementation, so it is opt-in only
        # (useful if you need to bound memory on a small GPU).
        processor_kwargs = dict(kwargs.pop("processor_kwargs", {}) or {})
        # VideoCollator already samples frames; without this the processor
        # resamples them again assuming 24 fps (64 frames -> 5, padded to 6)
        processor_kwargs.setdefault("do_sample_frames", False)
        if min_pixels is not None:
            processor_kwargs.setdefault("min_pixels", min_pixels)
        if max_pixels is not None:
            processor_kwargs.setdefault("max_pixels", max_pixels)
        if video_frame_budget:
            processor_kwargs.setdefault(
                "video_processor",
                load_frame_budget_video_processor(
                    model,
                    total_pixels=VIDEO_MAX_TOTAL_PIXELS,
                    revision=revision,
                    do_sample_frames=False,
                ),
            )
        kwargs["processor_kwargs"] = processor_kwargs

        super().__init__(
            model,
            revision=revision,
            device=device,
            fps=fps,
            max_frames=max_frames,
            num_frames=num_frames,
            **kwargs,
        )
        self.use_instructions = use_instructions

    def _collect_inputs(
        self,
        loader: DataLoader[BatchedInput],
        prefix: str,
    ) -> list[Any]:
        # The official format_mm_content puts media before text (video, image, text).
        # sentence-transformers>=6 keeps the dict's key order (5.x sorted the keys, i.e. image
        # before text but text before video), so fix the order here.
        # https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B/blob/main/scripts/qwen3_vl_reranker.py
        order = {"video": 0, "image": 1, "text": 2}
        return [
            {k: item[k] for k in sorted(item, key=lambda k: order.get(k, len(order)))}
            if isinstance(item, dict)
            else item
            for item in super()._collect_inputs(loader, prefix)
        ]

    @staticmethod
    def _normalize_instruction(instruction: str) -> str:
        instruction = instruction.strip()
        # Mirrors Qwen3VLEmbeddingWrapper: append a period if the instruction
        # does not already end in punctuation.
        if instruction and not unicodedata.category(instruction[-1]).startswith("P"):
            instruction += "."
        return instruction

    def get_task_instruction(
        self,
        task_metadata: TaskMetadata,
        prompt_type: PromptType | None = None,
    ) -> str:
        """Resolve the single task-level instruction passed to the reranker.

        Unlike a bi-encoder, the reranker consumes the query and the document in
        one forward pass, so it takes one instruction describing the relevance
        criterion rather than a per-side prefix. When a task defines separate
        query/document prompts, the query-side prompt is used.
        """
        prompt = task_metadata.prompt
        if isinstance(prompt, dict):
            prompt = prompt.get("query") or next(iter(prompt.values()), None)
        if not prompt:
            return DEFAULT_INSTRUCTION
        return self._normalize_instruction(prompt)

    def predict(
        self,
        inputs1: DataLoader[BatchedInput],
        inputs2: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> Array:
        if self.use_instructions:
            kwargs.setdefault(
                "prompt", self.get_task_instruction(task_metadata, prompt_type)
            )
        return super().predict(
            inputs1,
            inputs2,
            task_metadata=task_metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            prompt_type=prompt_type,
            **kwargs,
        )


qwen3_vl_reranker_2b = ModelMeta(
    loader=Qwen3VLRerankerWrapper,
    loader_kwargs={"video_frame_budget": True},
    name="Qwen/Qwen3-VL-Reranker-2B",
    model_type=["cross-encoder"],
    languages=QWEN3_VL_LANGUAGES,
    open_weights=True,
    revision="4bd860ac4f15ad1897a214615cccc700f8f71818",
    release_date="2026-01-07",
    modalities=["image", "text", "video"],
    n_parameters=2_127_532_032,
    n_embedding_parameters=311_164_928,
    memory_usage_mb=4058,
    embed_dim=None,  # not applicable: outputs a scalar relevance score
    license="apache-2.0",
    max_tokens=32768,
    reference="https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B",
    similarity_fn_name=None,
    framework=["Sentence Transformers", "PyTorch", "Transformers", "safetensors"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    citation=QWEN3_VL_EMBEDDING_CITATION,
    extra_requirements_groups=["multimodal-sbert"],
)

qwen3_vl_reranker_8b = ModelMeta(
    loader=Qwen3VLRerankerWrapper,
    # modules.json at this revision points module 1 to "1_CausalScoreHead", but the repo ships
    # "1_LogitScore/config.json" (renamed in the ST integration PR, path not updated), so every
    # sentence-transformers version builds LogitScore() without its token ids and fails.
    # https://huggingface.co/Qwen/Qwen3-VL-Reranker-8B/blob/b212dc8c91a8164aef1ea2de9c1a867611e75c04/modules.json
    loader_kwargs={
        "module_paths": {"1_CausalScoreHead": "1_LogitScore"},
        "video_frame_budget": True,
    },
    name="Qwen/Qwen3-VL-Reranker-8B",
    model_type=["cross-encoder"],
    languages=QWEN3_VL_LANGUAGES,
    open_weights=True,
    revision="b212dc8c91a8164aef1ea2de9c1a867611e75c04",
    release_date="2026-01-07",
    modalities=["image", "text", "video"],
    n_parameters=8_767_123_696,
    n_embedding_parameters=622_329_856,
    memory_usage_mb=16722,
    embed_dim=None,  # not applicable: outputs a scalar relevance score
    license="apache-2.0",
    max_tokens=32768,
    reference="https://huggingface.co/Qwen/Qwen3-VL-Reranker-8B",
    similarity_fn_name=None,
    framework=["Sentence Transformers", "PyTorch", "Transformers", "safetensors"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    citation=QWEN3_VL_EMBEDDING_CITATION,
    extra_requirements_groups=["multimodal-sbert"],
)
