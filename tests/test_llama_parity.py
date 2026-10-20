"""Our hand-written forward must match HuggingFace on tiny random checkpoints.

Runs on CPU in fp32 so it is exact enough to catch any layout or op-order bug,
and it runs in CI without a GPU.
"""

from __future__ import annotations

import pytest
import torch
from transformers import AutoModelForCausalLM, LlamaConfig, Qwen2Config

from safeinfer.attention.backend import AttentionMetadata
from safeinfer.attention.dense import DenseAttention
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


def _flatten(prompts: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor, AttentionMetadata]:
    input_ids = torch.tensor([t for p in prompts for t in p])
    positions = torch.cat([torch.arange(len(p)) for p in prompts])
    return input_ids, positions, AttentionMetadata.for_prefill([len(p) for p in prompts])


def test_logits_and_hidden_states_match_hf(models):
    hf, ours = models
    # Different lengths in one flattened batch; positions past the llama3
    # original context (64) exercise the scaled RoPE frequencies.
    gen = torch.Generator().manual_seed(1)
    prompts = [torch.randint(0, 256, (n,), generator=gen).tolist() for n in (5, 1, 100)]
    input_ids, positions, meta = _flatten(prompts)
    capture = 1

    hidden, captured = ours(input_ids, positions, DenseAttention(), meta, capture_layer=capture)
    logits = ours.compute_logits(hidden)

    start = 0
    for p in prompts:
        end = start + len(p)
        with torch.no_grad():
            ref = hf(torch.tensor([p]), output_hidden_states=True)
        torch.testing.assert_close(logits[start:end], ref.logits[0], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(captured[start:end], ref.hidden_states[capture + 1][0], atol=1e-5, rtol=1e-5)
        start = end


def test_greedy_decode_matches_hf_token_for_token(models):
    hf, ours = models
    prompt = [3, 17, 42, 7, 99]
    steps = 20

    with torch.no_grad():
        ref = hf.generate(
            torch.tensor([prompt]), max_new_tokens=steps, min_new_tokens=steps, do_sample=False, pad_token_id=0
        )[0, len(prompt) :].tolist()

    # Cache-free greedy loop: re-run the full prefix every step.
    tokens = list(prompt)
    for _ in range(steps):
        input_ids, positions, meta = _flatten([tokens])
        hidden, _ = ours(input_ids, positions, DenseAttention(), meta)
        tokens.append(int(ours.compute_logits(hidden[-1:]).argmax(-1)))

    assert tokens[len(prompt) :] == ref
