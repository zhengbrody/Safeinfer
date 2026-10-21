"""Scheduler policy tests, with no model: a fake step grows the cache and emits a dummy token."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from safeinfer.memory.block_manager import BlockManager
from safeinfer.sampling_params import SamplingParams
from safeinfer.scheduler.scheduler import Scheduler, SchedulerConfig
from safeinfer.scheduler.sequence import Sequence, SequenceStatus

TOKEN = 7


def _scheduler(num_blocks=64, block_size=4, eos=(), **cfg) -> Scheduler:
    return Scheduler(SchedulerConfig(**cfg), BlockManager(num_blocks, block_size), eos)


def _seq(seq_id: int, prompt_len: int, max_tokens: int, **params) -> Sequence:
    return Sequence(seq_id, list(range(1, prompt_len + 1)), SamplingParams(max_tokens=max_tokens, **params))


def _step(sched: Scheduler, token: int = TOKEN):
    """Schedule, play the model runner's part (grow the cache), and record sampled tokens."""
    out = sched.schedule()
    for s in out.scheduled:
        sched.block_manager.grow(s.seq.seq_id, s.num_tokens)
    sched.update(out.scheduled, [token if s.samples else None for s in out.scheduled])
    return out


def _plan(out) -> list[tuple[int, int]]:
    return [(s.seq.seq_id, s.num_tokens) for s in out.scheduled]


def test_new_requests_join_running_ones_mid_generation():
    sched = _scheduler(max_num_batched_tokens=64, max_model_len=64)
    sched.add(_seq(0, prompt_len=5, max_tokens=4))
    assert _plan(_step(sched)) == [(0, 5)]
    sched.add(_seq(1, prompt_len=3, max_tokens=2))
    # Seq 0 decodes one token while seq 1 prefills, in the same step.
    assert _plan(_step(sched)) == [(0, 1), (1, 3)]
    assert _plan(_step(sched)) == [(0, 1), (1, 1)]
    # Seq 1 hit max_tokens and left; seq 0 continues alone.
    assert [s.seq_id for s in sched.running] == [0]
    assert sched.block_manager.num_tokens(1) == 0


def test_token_budget_defers_prompts_without_chunking():
    sched = _scheduler(max_num_batched_tokens=10, max_model_len=10)
    for i, n in enumerate((6, 6, 2)):
        sched.add(_seq(i, prompt_len=n, max_tokens=1))
    # Seq 1 does not fit the remaining 4 tokens; FCFS means seq 2 waits behind it.
    assert _plan(_step(sched)) == [(0, 6)]


def test_chunked_prefill_splits_prompt_and_shares_steps_with_decode():
    sched = _scheduler(max_num_batched_tokens=4, max_model_len=32, enable_chunked_prefill=True)
    sched.add(_seq(0, prompt_len=2, max_tokens=5))
    sched.add(_seq(1, prompt_len=9, max_tokens=1))
    assert _plan(_step(sched)) == [(0, 2), (1, 2)]
    assert _plan(_step(sched)) == [(0, 1), (1, 3)]
    out = _step(sched)
    assert _plan(out) == [(0, 1), (1, 3)]
    assert not out.scheduled[1].samples  # 8 of 9 prompt tokens done
    out = _step(sched)
    assert _plan(out) == [(0, 1), (1, 1)] and out.scheduled[1].samples
    assert not sched.waiting and [s.seq_id for s in sched.running] == [0]


def test_max_num_seqs_caps_running():
    sched = _scheduler(max_num_seqs=2, max_num_batched_tokens=64, max_model_len=64)
    for i in range(3):
        sched.add(_seq(i, prompt_len=2, max_tokens=3))
    assert [i for i, _ in _plan(_step(sched))] == [0, 1]
    assert [i for i, _ in _plan(_step(sched))] == [0, 1]


def test_preemption_evicts_newest_and_recomputes_it_first():
    # 4 blocks of 4 tokens. Two 7-token prompts take 2 blocks each; the 9th token needs a 5th block.
    sched = _scheduler(num_blocks=4, block_size=4, max_num_batched_tokens=16, max_model_len=16)
    sched.add(_seq(0, prompt_len=7, max_tokens=6))
    sched.add(_seq(1, prompt_len=7, max_tokens=6))
    sched.add(_seq(2, prompt_len=1, max_tokens=1))
    assert _plan(_step(sched)) == [(0, 7), (1, 7)]  # seq 2 waits for blocks
    assert _plan(_step(sched)) == [(0, 1), (1, 1)]  # both fill their second block

    out = _step(sched)
    assert _plan(out) == [(0, 1)]
    assert [s.seq_id for s in out.preempted] == [1]
    victim = out.preempted[0]
    assert victim.status is SequenceStatus.WAITING and victim.num_computed_tokens == 0
    assert victim.output_token_ids == [TOKEN, TOKEN]  # generated tokens are kept
    assert [s.seq_id for s in sched.waiting] == [1, 2]  # ahead of the request that never ran
    assert sched.num_preemptions == 1

    for _ in range(3):  # seq 0 runs alone: the 3 blocks seq 1 needs back are not free
        assert _plan(_step(sched)) == [(0, 1)]
    assert sched.running == []
    # Seq 1 resumes by recomputing prompt + its 2 outputs in one prefill, then seq 2 finally runs.
    assert _plan(_step(sched)) == [(1, 9), (2, 1)]


def test_stop_conditions():
    sched = _scheduler(eos=(TOKEN,), max_num_batched_tokens=32, max_model_len=32)
    sched.add(_seq(0, prompt_len=2, max_tokens=5))
    sched.add(_seq(1, prompt_len=2, max_tokens=3, ignore_eos=True))
    sched.add(_seq(2, prompt_len=2, max_tokens=5, ignore_eos=True, stop_token_ids=(9,)))
    a, b, c = list(sched.waiting)
    _step(sched)
    assert (a.finish_reason, a.output_token_ids) == ("stop", [TOKEN])  # EOS kept in the output
    assert b.finish_reason is None and c.finish_reason is None
    _step(sched, token=9)
    assert c.finish_reason == "stop" and b.finish_reason is None
    _step(sched)
    assert b.finish_reason == "length" and b.output_token_ids == [TOKEN, 9, TOKEN]
    assert not sched.has_unfinished() and sched.block_manager.num_free_blocks == 64


def test_config_validation():
    with pytest.raises(ValueError, match="max_num_batched_tokens"):
        SchedulerConfig(max_num_batched_tokens=512, max_model_len=2048)
    SchedulerConfig(max_num_batched_tokens=512, max_model_len=2048, enable_chunked_prefill=True)
    # The pool must hold one max-length sequence, or a lone runner could never grow.
    with pytest.raises(ValueError, match="cannot hold"):
        _scheduler(num_blocks=3, block_size=4, max_num_batched_tokens=16, max_model_len=16)
    sched = _scheduler(max_num_batched_tokens=16, max_model_len=16)
    with pytest.raises(ValueError, match="max_model_len"):
        sched.add(_seq(0, prompt_len=12, max_tokens=5))
    with pytest.raises(ValueError, match="empty"):
        sched.add(_seq(0, prompt_len=0, max_tokens=1))


@settings(max_examples=150, deadline=None)
@given(
    requests=st.lists(
        st.tuples(st.integers(1, 20), st.integers(1, 12), st.integers(0, 15)),  # prompt, output, arrival step
        min_size=1,
        max_size=12,
    ),
    num_blocks=st.integers(8, 24),
    block_size=st.sampled_from([1, 2, 4]),
    max_num_seqs=st.integers(1, 6),
    chunked=st.booleans(),
    budget=st.integers(1, 40),
)
def test_random_workloads_finish_within_limits(requests, num_blocks, block_size, max_num_seqs, chunked, budget):
    max_model_len = 32
    if num_blocks * block_size < max_model_len:
        num_blocks = -(-max_model_len // block_size)
    if not chunked:
        budget = max(budget, max_model_len)
    sched = _scheduler(
        num_blocks=num_blocks,
        block_size=block_size,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=budget,
        max_model_len=max_model_len,
        enable_chunked_prefill=chunked,
    )
    bm = sched.block_manager
    seqs = [_seq(i, p, o) for i, (p, o, _) in enumerate(requests)]
    arrivals = sorted(range(len(seqs)), key=lambda i: requests[i][2])

    step = 0
    while arrivals or sched.has_unfinished():
        while arrivals and requests[arrivals[0]][2] <= step:
            sched.add(seqs[arrivals.pop(0)])
        out = _step(sched)
        step += 1
        assert sum(s.num_tokens for s in out.scheduled) <= budget
        assert len(sched.running) <= max_num_seqs
        assert len({s.seq.seq_id for s in out.scheduled}) == len(out.scheduled)
        bm.check_invariants()
        for s in sched.running:
            assert bm.num_tokens(s.seq_id) == s.num_computed_tokens > 0
        for s in sched.waiting:
            assert bm.num_tokens(s.seq_id) == 0
        # Generous bound: every token is (re)computed in bounded time; catches livelock.
        assert step < 20 + 200 * len(seqs)

    assert all(len(s.output_token_ids) == s.params.max_tokens and s.finish_reason == "length" for s in seqs)
    assert bm.num_free_blocks == num_blocks
