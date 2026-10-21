"""Shared fixtures: tiny random HF checkpoints and our model loaded from them."""

from __future__ import annotations

import pytest
import torch
from transformers import AutoModelForCausalLM, LlamaConfig, Qwen2Config

from safeinfer.model.loader import load_model

TINY = {
    "vocab_size": 256,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 3,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,  # GQA
    "max_position_embeddings": 512,
}
LLAMA3_ROPE = {
    "rope_type": "llama3",
    "rope_theta": 500000.0,
    "factor": 8.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 64,
}

VARIANTS = {
    "llama": lambda: LlamaConfig(**TINY),
    "llama3-rope-tied": lambda: LlamaConfig(**TINY, rope_parameters=LLAMA3_ROPE, tie_word_embeddings=True),
    "qwen2-bias": lambda: Qwen2Config(**TINY, tie_word_embeddings=True),
}


@pytest.fixture(params=list(VARIANTS))
def models(request, tmp_path):
    torch.manual_seed(0)
    hf = AutoModelForCausalLM.from_config(VARIANTS[request.param]()).float().eval()
    # HF zero-inits biases and ones-inits norms; randomize everything so a
    # parameter that fails to load (or loads into the wrong slot) changes outputs.
    with torch.no_grad():
        for p in hf.parameters():
            p.normal_(0.0, 0.1)
    hf.save_pretrained(tmp_path)
    ours = load_model(tmp_path, dtype=torch.float32, device="cpu")
    return hf, ours
