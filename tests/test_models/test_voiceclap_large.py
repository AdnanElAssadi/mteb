from __future__ import annotations

import json
from pathlib import Path

import mteb


def test_voiceclap_large_points_at_loadable_repo():
    """The pinned repo must ship what sentence-transformers needs and merged weights.

    VoiceNet/voiceclap-large has no 1_Pooling config or chat template, and stores
    PEFT-wrapped keys that load as random weights.
    """
    from huggingface_hub import hf_hub_download, list_repo_files

    meta = mteb.get_model_meta("laion/voiceclap-large")
    files = list_repo_files(meta.name, revision=meta.revision)
    assert "1_Pooling/config.json" in files
    assert "additional_chat_templates/sentence_transformers.jinja" in files

    index = json.loads(
        Path(
            hf_hub_download(
                meta.name, "model.safetensors.index.json", revision=meta.revision
            )
        ).read_text(encoding="utf-8")
    )
    assert not any(
        ".base_layer." in key or ".lora_" in key for key in index["weight_map"]
    )
    # bf16 weights: 2 bytes per parameter
    assert meta.n_parameters == index["metadata"]["total_size"] // 2
