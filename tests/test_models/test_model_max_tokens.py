from __future__ import annotations

import re
from pathlib import Path

import pytest

import mteb


@pytest.mark.parametrize(
    "model_name",
    [
        "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "Qwen/Qwen3-Omni-30B-A3B-Thinking",
        "Qwen/Qwen3-Omni-30B-A3B-Captioner",
    ],
)
def test_qwen3_omni_max_tokens_matches_hub_config(model_name: str):
    from transformers import PretrainedConfig

    meta = mteb.get_model_meta(model_name)
    config, _ = PretrainedConfig.get_config_dict(model_name, revision=meta.revision)
    text_config = config["thinker_config"]["text_config"]
    assert meta.max_tokens == text_config["max_position_embeddings"]


@pytest.mark.parametrize(
    "model_name", ["OpenGVLab/ViCLIP-L-14-hf", "OpenGVLab/ViCLIP-B-16-hf"]
)
def test_viclip_max_tokens_matches_remote_code(model_name: str):
    from huggingface_hub import hf_hub_download

    meta = mteb.get_model_meta(model_name)
    path = hf_hub_download(model_name, "viclip.py", revision=meta.revision)
    source = Path(path).read_text(encoding="utf-8")
    max_txt_l = re.search(r"self\.max_txt_l = (\d+)", source)
    assert max_txt_l is not None
    assert meta.max_tokens == int(max_txt_l.group(1))
