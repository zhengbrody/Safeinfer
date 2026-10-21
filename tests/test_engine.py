"""Continuous batching must not change what each request generates.

Requests arrive mid-generation, the KV pool is small enough to force
preemption, and chunked prefill splits prompts; greedy output must still
equal HF generating each request alone, and seeded sampling must not depend
on batching.
"""

from __future__ import annotations

import pytest
import torch

from safeinfer.engine.engine import LLMEngine
from safeinfer.engine.model_runner import ModelRunner
from safeinfer.sampling_params import SamplingParams
from safeinfer.scheduler.scheduler import SchedulerConfig

BLOCK_SIZE = 4
PROMPT_LENS = (5, 1, 13, 8, 3, 10)
MAX_TOKENS = (12, 3, 7, 9, 14, 5)


def _prompt(n: int, seed: int) -> list[int]:
    return torch.randint(0, 256, (n,), generator=torch.Generator().manual_seed(seed)).tolist()


def _engine(model, num_blocks: int, chunked: bool, budget: int = 32) -> LLMEngine:
    config = SchedulerConfig(
        max_num_seqs=4,
        max_num_batched_tokens=budget,
        max_model_len=32,
        enable_chunked_prefill=chunked,
    )
    return LLMEngine(ModelRunner(model, num_blocks=num_blocks, block_size=BLOCK_SIZE), config)


def _run_staggered(engine: LLMEngine, prompts, params, join_every: int = 2) -> list[list[int]]:
    """Add one request every `join_every` steps, so requests join a batch already decoding."""
    outputs: dict[int, list[int]] = {}
    queue = list(zip(prompts, params, strict=True))
    ids: list[int] = []
    step = 0
    while queue or engine.has_unfinished_requests():
        if queue and step % join_every == 0:
            ids.append(engine.add_request(*queue.pop(0)))
            outputs[ids[-1]] = []
        for o in engine.step():
            outputs[o.seq_id].append(o.token_id)
        step += 1
    return [outputs[i] for i in ids]


# Tight pool (10 blocks = 40 slots for up to 4 sequences) forces preemption; the
# small budget with chunking splits the 13-token prompt across steps.
@pytest.mark.parametrize(("chunked", "budget"), [(False, 32), (True, 6)])
def test_continuous_greedy_matches_hf_under_preemption(models, chunked, budget):
    hf, ours = models
    prompts = [_prompt(n, seed=n) for n in PROMPT_LENS]
    params = [SamplingParams(max_tokens=n, ignore_eos=True) for n in MAX_TOKENS]
    engine = _engine(ours, num_blocks=10, chunked=chunked, budget=budget)

    got = _run_staggered(engine, prompts, params)

    for p, n, out in zip(prompts, MAX_TOKENS, got, strict=True):
        with torch.no_grad():
            ref = hf.generate(torch.tensor([p]), max_new_tokens=n, min_new_tokens=n, do_sample=False, pad_token_id=0)
        assert out == ref[0, len(p) :].tolist()
    assert engine.num_preemptions > 0
    bm = engine.runner.block_manager
    assert bm.num_free_blocks == 10
    bm.check_invariants()


def test_seeded_sampling_is_independent_of_batching_and_preemption(models):
    _, ours = models
    prompts = [_prompt(n, seed=100 + n) for n in PROMPT_LENS]
    params = [SamplingParams(max_tokens=n, temperature=0.8, top_p=0.9, seed=i) for i, n in enumerate(MAX_TOKENS)]

    crowded = _engine(ours, num_blocks=10, chunked=True, budget=6)
    together = _run_staggered(crowded, prompts, params)
    assert crowded.num_preemptions > 0
    for p, sp, out in zip(prompts, params, together, strict=True):
        alone = _engine(ours, num_blocks=64, chunked=False).generate([p], [sp])[0]
        assert out == alone


def test_stop_token_ends_generation_and_frees_blocks(models):
    _, ours = models
    prompt = _prompt(6, seed=7)
    engine = _engine(ours, num_blocks=16, chunked=False)
    full = engine.generate([prompt], [SamplingParams(max_tokens=10, ignore_eos=True)])[0]
    cut = max(i for i, t in enumerate(full) if t not in full[:i])  # latest first-occurrence: a later stop

    engine.add_request(prompt, SamplingParams(max_tokens=10, ignore_eos=True, stop_token_ids=(full[cut],)))
    outputs = []
    while engine.has_unfinished_requests():
        outputs += engine.step()
    assert [o.token_id for o in outputs] == full[: cut + 1]
    assert outputs[-1].finished and outputs[-1].finish_reason == "stop"
    assert engine.runner.block_manager.num_free_blocks == 16
