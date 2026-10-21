"""The paged KV path must reproduce cache-free results and HF generation.

Block size 4 keeps every sequence crossing several block boundaries, and the
pool is small enough that freed blocks get reused across sequences.
"""

from __future__ import annotations

import pytest
import torch

from safeinfer.attention.backend import AttentionMetadata
from safeinfer.attention.dense import DenseAttention
from safeinfer.engine.model_runner import ModelRunner, SeqStep
from safeinfer.engine.static_batch import generate_static
from safeinfer.memory.block_manager import OutOfBlocksError

BLOCK_SIZE = 4


def _prompt(n: int, seed: int) -> list[int]:
    return torch.randint(0, 256, (n,), generator=torch.Generator().manual_seed(seed)).tolist()


def _dense_last_logits(model, tokens: list[int]) -> torch.Tensor:
    meta = AttentionMetadata.for_prefill([len(tokens)])
    hidden, _ = model(torch.tensor(tokens), torch.arange(len(tokens)), DenseAttention(), meta)
    return model.compute_logits(hidden[-1:])[0]


def test_batched_greedy_matches_hf(models):
    hf, ours = models
    prompts = [_prompt(n, seed=n) for n in (5, 1, 13, 8)]
    budgets = [12, 3, 7, 1]
    runner = ModelRunner(ours, num_blocks=24, block_size=BLOCK_SIZE)

    outputs = generate_static(runner, prompts, budgets)

    for p, n, got in zip(prompts, budgets, outputs, strict=True):
        with torch.no_grad():
            ref = hf.generate(torch.tensor([p]), max_new_tokens=n, min_new_tokens=n, do_sample=False, pad_token_id=0)
        assert got == ref[0, len(p) :].tolist()
    # Everything was freed and the pool is intact.
    assert runner.block_manager.num_free_blocks == 24
    runner.block_manager.check_invariants()


def test_chunked_prefill_and_mixed_batch_match_dense(models):
    _, ours = models
    long, short = _prompt(11, seed=1), _prompt(3, seed=2)
    runner = ModelRunner(ours, num_blocks=16, block_size=BLOCK_SIZE)

    # Step 1: first chunk of `long` and all of `short`, together.
    runner.step([SeqStep(0, long[:6]), SeqStep(1, short)])
    # Step 2: the rest of `long` (chunked prefill) alongside a decode token for `short`.
    out = runner.step([SeqStep(0, long[6:]), SeqStep(1, [42])])

    torch.testing.assert_close(out.logits[0], _dense_last_logits(ours, long), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out.logits[1], _dense_last_logits(ours, short + [42]), atol=1e-5, rtol=1e-5)


def test_captured_activation_is_last_token_of_each_sequence(models):
    _, ours = models
    prompts = [_prompt(6, seed=3), _prompt(9, seed=4)]
    runner = ModelRunner(ours, num_blocks=16, block_size=BLOCK_SIZE, capture_layers=(1, 2))
    out = runner.step([SeqStep(i, p) for i, p in enumerate(prompts)])

    for i, p in enumerate(prompts):
        meta = AttentionMetadata.for_prefill([len(p)])
        _, ref = ours(torch.tensor(p), torch.arange(len(p)), DenseAttention(), meta, capture_layers=(1, 2))
        torch.testing.assert_close(out.captured[i], ref[-1], atol=1e-5, rtol=1e-5)


def test_step_is_atomic_when_cache_is_full(models):
    _, ours = models
    runner = ModelRunner(ours, num_blocks=3, block_size=BLOCK_SIZE)
    runner.step([SeqStep(0, _prompt(8, seed=5))])  # 2 of 3 blocks

    with pytest.raises(OutOfBlocksError):
        runner.step([SeqStep(1, _prompt(2, seed=6)), SeqStep(2, _prompt(2, seed=7))])
    assert runner.block_manager.num_tokens(1) == 0
    assert runner.block_manager.num_free_blocks == 1
