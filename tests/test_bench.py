from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from transformers import AutoModelForCausalLM, LlamaConfig

from safeinfer.bench.hf_baseline import run_static_batching
from safeinfer.bench.metrics import RequestTiming, summarize
from safeinfer.bench.workload import build_workload, fixed_lengths, poisson_arrival_times


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

    timings = run_static_batching(model, requests, max_batch_size=4)

    assert sorted(t.request_id for t in timings) == list(range(6))
    for t in timings:
        assert t.arrival_time <= t.first_token_time <= t.finish_time
    # Requests 4 and 5 form the second batch, so they start after the first batch ends.
    first_batch_end = max(t.finish_time for t in timings if t.request_id < 4)
    second_batch_start = min(t.first_token_time for t in timings if t.request_id >= 4)
    assert second_batch_start > first_batch_end
