from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from transformers import AutoModelForCausalLM, LlamaConfig

from safeinfer.bench.engine_static import engine_batch_runner, engine_can_admit
from safeinfer.bench.hf_baseline import hf_batch_runner
from safeinfer.bench.metrics import RequestTiming, summarize
from safeinfer.bench.static_replay import replay_static_batching
from safeinfer.bench.workload import build_workload, fixed_lengths, poisson_arrival_times
from safeinfer.engine.model_runner import ModelRunner


def test_poisson_arrivals_have_requested_rate():
    times = poisson_arrival_times(20000, rate=5.0, rng=np.random.default_rng(0))
    assert np.all(np.diff(times) >= 0)
    assert len(times) / times[-1] == pytest.approx(5.0, rel=0.05)


def test_infinite_rate_is_offline_mode():
    assert np.all(poisson_arrival_times(10, math.inf, np.random.default_rng(0)) == 0)


def test_workload_is_deterministic_and_respects_lengths():
    lengths = [(7, 3), (2, 9)]
    a = build_workload(lengths, rate=2.0, vocab_size=256, seed=1)
    b = build_workload(lengths, rate=2.0, vocab_size=256, seed=1)
    assert a == b
    assert [(r.prompt_len, r.output_len) for r in a] == lengths
    assert all(0 <= t < 256 for r in a for t in r.prompt_token_ids)


def test_summarize_hand_computed():
    timings = [
        RequestTiming(0, prompt_len=10, output_len=5, arrival_time=0.0, first_token_time=0.1, finish_time=0.5),
        RequestTiming(1, prompt_len=10, output_len=1, arrival_time=0.5, first_token_time=0.8, finish_time=0.8),
    ]
    s = summarize(timings)
    assert s["duration_s"] == pytest.approx(0.8)
    assert s["output_token_throughput"] == pytest.approx(6 / 0.8)
    assert s["mean_ttft_ms"] == pytest.approx(200.0)
    # Single-token outputs have no TPOT, so only request 0 contributes: 0.4s / 4 tokens.
    assert s["mean_tpot_ms"] == pytest.approx(100.0)


def test_static_batching_timings_are_consistent():
    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=256,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    model = AutoModelForCausalLM.from_config(cfg).eval()
    requests = build_workload(fixed_lengths(5, prompt_len=6, output_len=4) + [(3, 9)], math.inf, 256)

    timings = replay_static_batching(requests, hf_batch_runner(model), max_batch_size=4)

    assert sorted(t.request_id for t in timings) == list(range(6))
    for t in timings:
        assert t.arrival_time <= t.first_token_time <= t.finish_time
    # Requests 4 and 5 form the second batch, so they start after the first batch ends.
    first_batch_end = max(t.finish_time for t in timings if t.request_id < 4)
    second_batch_start = min(t.first_token_time for t in timings if t.request_id >= 4)
    assert second_batch_start > first_batch_end


def test_engine_static_admits_by_kv_capacity(models):
    _, ours = models
    # Each request needs ceil((6 + 4) / 4) = 3 blocks; 7 blocks fit two at a time.
    runner = ModelRunner(ours, num_blocks=7, block_size=4)
    requests = build_workload(fixed_lengths(5, prompt_len=6, output_len=4), math.inf, 256)

    timings = replay_static_batching(requests, engine_batch_runner(runner), 32, engine_can_admit(runner))

    assert sorted(t.request_id for t in timings) == list(range(5))
    starts = sorted({t.first_token_time for t in timings})
    assert len(starts) == 3  # batches of 2, 2, 1
    assert runner.block_manager.num_free_blocks == 7


def test_replay_rejects_request_larger_than_server(models):
    _, ours = models
    runner = ModelRunner(ours, num_blocks=2, block_size=4)
    requests = build_workload([(6, 4)], math.inf, 256)
    with pytest.raises(RuntimeError, match="never be admitted"):
        replay_static_batching(requests, engine_batch_runner(runner), 32, engine_can_admit(runner))


def test_kv_utilization_hand_computed():
    from safeinfer.bench.kv_utilization import analyze

    # One request: prompt 3, output 2 -> lengths 4, 5 over its life.
    rows = analyze([(3, 2)], max_model_len=10, block_sizes=[4], budget_tokens=100)
    contiguous, paged = rows
    assert contiguous["utilization"] == pytest.approx((4 + 5) / (10 + 10))
    assert contiguous["concurrency"] == 10
    # Paged reserves 4 slots at length 4, then 8 at length 5.
    assert paged["utilization"] == pytest.approx((4 + 5) / (4 + 8))
    assert paged["concurrency"] == 100 // 8
