"""Parity against HF on a real checkpoint (fp32, CPU).

Skipped unless SAFEINFER_TEST_MODEL names a hub id or local directory, e.g.

    SAFEINFER_TEST_MODEL=meta-llama/Llama-3.2-1B-Instruct pytest tests/test_real_checkpoint.py

A 1B model needs ~10 GB of RAM (ours + HF, both fp32). This catches what tiny
random models cannot: real llama3 RoPE scaling at 500k theta, tied 128k-vocab
embeddings, and the chat template's special tokens flowing through the model.
"""

from __future__ import annotations

import os

import pytest
import torch

from safeinfer.attention.backend import AttentionMetadata
from safeinfer.attention.dense import DenseAttention
from safeinfer.engine.model_runner import ModelRunner
from safeinfer.engine.static_batch import generate_static
from safeinfer.model.loader import load_model, resolve_model_dir

MODEL = os.environ.get("SAFEINFER_TEST_MODEL")
pytestmark = pytest.mark.skipif(not MODEL, reason="set SAFEINFER_TEST_MODEL to run")


@pytest.fixture(scope="module")
def setup():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_dir = resolve_model_dir(MODEL)
    tok = AutoTokenizer.from_pretrained(model_dir)
    hf = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).eval()
    ours = load_model(model_dir, dtype=torch.float32, device="cpu")
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": "Explain paged attention in one sentence."}],
        add_generation_prompt=True,
        return_dict=False,  # transformers 5 defaults to a BatchEncoding
    )
    return hf, ours, list(prompt)


def test_prefill_logits_match(setup):
    hf, ours, prompt = setup
    meta = AttentionMetadata.for_prefill([len(prompt)])
    hidden, captured = ours(torch.tensor(prompt), torch.arange(len(prompt)), DenseAttention(), meta, capture_layer=7)
    with torch.no_grad():
        ref = hf(torch.tensor([prompt]), output_hidden_states=True)

    torch.testing.assert_close(ours.compute_logits(hidden), ref.logits[0], atol=1e-3, rtol=1e-4)
    torch.testing.assert_close(captured, ref.hidden_states[8][0], atol=1e-3, rtol=1e-4)


def test_greedy_generation_matches(setup):
    hf, ours, prompt = setup
    steps = 24
    with torch.no_grad():
        ref = hf.generate(torch.tensor([prompt]), max_new_tokens=steps, min_new_tokens=steps, do_sample=False)
    ref = ref[0, len(prompt) :].tolist()

    tokens = list(prompt)
    for _ in range(steps):
        meta = AttentionMetadata.for_prefill([len(tokens)])
        hidden, _ = ours(torch.tensor(tokens), torch.arange(len(tokens)), DenseAttention(), meta)
        tokens.append(int(ours.compute_logits(hidden[-1:]).argmax(-1)))

    assert tokens[len(prompt) :] == ref


def test_paged_batched_generation_matches(setup):
    """Two prompts of different lengths decoded together through the paged cache."""
    hf, ours, prompt = setup
    prompts = [prompt, prompt[:20]]
    steps = 16
    runner = ModelRunner(ours, num_blocks=16, block_size=16)

    outputs = generate_static(runner, prompts, [steps, steps])

    for p, got in zip(prompts, outputs, strict=True):
        with torch.no_grad():
            ref = hf.generate(torch.tensor([p]), max_new_tokens=steps, min_new_tokens=steps, do_sample=False)
        assert got == ref[0, len(p) :].tolist()


def test_continuous_batching_with_preemption_matches(setup):
    """Staggered arrivals, chunked prefill, and a pool too small for all three: outputs unchanged.

    The first request keeps EOS enabled, so it also checks that we stop exactly where HF does.
    """
    from safeinfer.engine.engine import LLMEngine
    from safeinfer.sampling_params import SamplingParams
    from safeinfer.scheduler.scheduler import SchedulerConfig

    hf, ours, prompt = setup
    prompts = [prompt, prompt[:20], prompt[:30]]
    params = [
        SamplingParams(max_tokens=48),
        SamplingParams(max_tokens=24, ignore_eos=True),
        SamplingParams(max_tokens=32, ignore_eos=True),
    ]
    config = SchedulerConfig(max_num_batched_tokens=32, max_model_len=128, enable_chunked_prefill=True)
    engine = LLMEngine(ModelRunner(ours, num_blocks=10, block_size=16), config)

    outputs: dict[int, list[int]] = {}
    for p, sp in zip(prompts, params, strict=True):
        outputs[engine.add_request(p, sp)] = []
        for _ in range(3):  # the next request arrives while this one is mid-flight
            for o in engine.step():
                outputs[o.seq_id].append(o.token_id)
    while engine.has_unfinished_requests():
        for o in engine.step():
            outputs[o.seq_id].append(o.token_id)

    assert engine.num_preemptions > 0
    for p, sp, got in zip(prompts, params, outputs.values(), strict=True):
        n = sp.max_tokens
        kwargs = {} if not sp.ignore_eos else {"min_new_tokens": n}
        with torch.no_grad():
            ref = hf.generate(torch.tensor([p]), max_new_tokens=n, do_sample=False, **kwargs)
        assert got == ref[0, len(p) :].tolist()
