"""Probe/monitor unit tests, and the monitor inside the continuous-batching engine.

The engine tests use a random probe: what matters here is plumbing (each
output position is scored exactly once, from the right activation, even
under chunked prefill and preemption), not whether the probe means anything.
"""

from __future__ import annotations

import pytest
import torch

from safeinfer.attention.backend import AttentionMetadata
from safeinfer.attention.dense import DenseAttention
from safeinfer.engine.engine import LLMEngine
from safeinfer.engine.model_runner import ModelRunner
from safeinfer.safety.monitor import SafetyMonitor
from safeinfer.safety.probe import LinearProbe
from safeinfer.sampling_params import SamplingParams
from safeinfer.scheduler.scheduler import SchedulerConfig

LAYER = 1
HIDDEN = 64  # conftest TINY hidden_size


def _probe(threshold: float = 0.5, layer: int = LAYER, seed: int = 0) -> LinearProbe:
    w = torch.randn(HIDDEN, generator=torch.Generator().manual_seed(seed))
    return LinearProbe(layer, w, 0.0, threshold)


def test_standardization_folds_into_weights(tmp_path):
    g = torch.Generator().manual_seed(0)
    x, mean, std, coef = (
        torch.randn(5, 8, generator=g),
        torch.randn(8, generator=g),
        torch.rand(8) + 0.5,
        torch.randn(8),
    )
    probe = LinearProbe.from_standardized(3, coef, 0.25, mean, std, threshold=0.7)
    torch.testing.assert_close(probe.score(x), torch.sigmoid(((x - mean) / std) @ coef + 0.25))

    probe.save(tmp_path / "p.safetensors")
    loaded = LinearProbe.load(tmp_path / "p.safetensors")
    assert (loaded.layer, loaded.bias, loaded.threshold) == (3, probe.bias, 0.7)
    torch.testing.assert_close(loaded.score(x), probe.score(x))


def test_consecutive_rule_is_sticky_and_needs_a_run():
    probe = LinearProbe(0, torch.ones(1), 0.0, threshold=0.5)  # score >= 0.5 iff x >= 0
    mon = SafetyMonitor(probe, consecutive=2)
    flags = []
    for pos, x in enumerate([1.0, -1.0, 1.0, 1.0, -1.0]):
        (sig,) = mon.observe([7], [pos], {0: torch.tensor([[x]])})
        flags.append(sig.flagged)
    # A single hit (pos 0) is not enough; pos 2-3 make a run of two; the flag then stays set.
    assert flags == [False, False, False, True, True]


def test_positions_must_advance_by_one():
    mon = SafetyMonitor(LinearProbe(0, torch.ones(1), 0.0, 0.5))
    mon.observe([1], [0], {0: torch.zeros(1, 1)})
    with pytest.raises(ValueError, match="expected 1"):
        mon.observe([1], [0], {0: torch.zeros(1, 1)})  # rescoring a position (e.g. after recompute)
    mon.release(1)
    mon.observe([1], [0], {0: torch.zeros(1, 1)})  # a released id starts fresh


def test_input_probe_scores_only_position_zero():
    probe = LinearProbe(0, torch.ones(1), 0.0, 0.5)
    mon = SafetyMonitor(probe, input_probe=LinearProbe(2, torch.ones(1), 0.0, 0.5))
    assert mon.layers == (0, 2)
    acts = {0: torch.zeros(2, 1), 2: torch.zeros(2, 1)}
    a, b = mon.observe([1, 2], [0, 0], acts)
    assert a.input_score == b.input_score == 0.5
    a, _ = mon.observe([1, 2], [1, 1], acts)
    assert a.input_score is None


def _engine(model, monitor, num_blocks=10, chunked=True, budget=6):
    config = SchedulerConfig(
        max_num_seqs=4, max_num_batched_tokens=budget, max_model_len=32, enable_chunked_prefill=chunked
    )
    runner = ModelRunner(model, num_blocks=num_blocks, block_size=4, capture_layers=(0, LAYER))
    return LLMEngine(runner, config, monitor)


def _prompts():
    return [torch.randint(0, 256, (n,), generator=torch.Generator().manual_seed(n)).tolist() for n in (5, 13, 8, 10)]


def _run(engine, prompts, max_tokens=12):
    """Add one request every other step; collect per-request tokens and safety signals."""
    tokens, signals = {}, {}
    queue = list(prompts)
    step = 0
    while queue or engine.has_unfinished_requests():
        if queue and step % 2 == 0:
            sid = engine.add_request(queue.pop(0), SamplingParams(max_tokens=max_tokens, ignore_eos=True))
            tokens[sid], signals[sid] = [], []
        for o in engine.step():
            signals[o.seq_id].append(o.safety)
            if o.token_id is not None:
                tokens[o.seq_id].append(o.token_id)
        step += 1
    return list(tokens.values()), list(signals.values())


def test_engine_scores_every_position_from_the_right_activation(models):
    _, ours = models
    probe = _probe()
    prompts = _prompts()
    engine = _engine(ours, SafetyMonitor(probe))

    tokens, signals = _run(engine, prompts)

    assert engine.scheduler.num_preemptions > 0  # the monitor survived recompute without rescoring
    for p, toks, sigs in zip(prompts, tokens, signals, strict=True):
        assert len(sigs) == len(toks) == 12
        for t, sig in enumerate(sigs):
            # Position t is predicted from the last token of prompt + first t outputs.
            ctx = p + toks[:t]
            meta = AttentionMetadata.for_prefill([len(ctx)])
            _, ref = ours(torch.tensor(ctx), torch.arange(len(ctx)), DenseAttention(), meta, capture_layers=(LAYER,))
            assert sig.score == pytest.approx(probe.score(ref[-1, 0]).item(), abs=1e-5)


def test_flag_mode_does_not_change_outputs_and_stop_mode_withholds_flagged_token(models):
    _, ours = models
    prompts = _prompts()
    plain, _ = _run(_engine(ours, None), prompts)
    flagged, signals = _run(_engine(ours, SafetyMonitor(_probe())), prompts)
    assert flagged == plain

    # Pick a threshold so request 1 first reaches two consecutive hits at a known position.
    scores = [s.score for s in signals[1]]
    pairs = [min(scores[t - 1], scores[t]) for t in range(1, len(scores))]
    threshold = sorted(pairs)[-1]  # only the best adjacent pair clears it
    first = 1 + pairs.index(threshold)

    stop_mon = SafetyMonitor(_probe(threshold), consecutive=2, stop=True)
    engine = _engine(ours, stop_mon)
    stopped, stop_signals = _run(engine, prompts)

    assert stopped[1] == plain[1][:first]  # token `first` was never emitted
    assert stop_signals[1][-1].flagged and len(stop_signals[1]) == first + 1
    for i in (0, 2, 3):  # other requests unaffected unless they also cross the threshold
        flagged_at = next((t for t, s in enumerate(stop_signals[i]) if s.flagged), None)
        assert stopped[i] == plain[i][:flagged_at]
    assert engine.runner.block_manager.num_free_blocks == 10
    assert stop_mon._state == {}


def test_engine_requires_captured_probe_layer(models):
    _, ours = models
    runner = ModelRunner(ours, num_blocks=10, block_size=4)
    with pytest.raises(ValueError, match="capture layers"):
        LLMEngine(runner, SchedulerConfig(max_model_len=32), SafetyMonitor(_probe()))
