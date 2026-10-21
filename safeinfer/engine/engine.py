"""Continuous-batching engine: scheduler + model runner + sampler, one step at a time.

Each `step()` runs one forward over whatever the scheduler picked (new
prompts, prompt chunks, and decode tokens mixed), samples a token for every
sequence that reached its end, and returns those tokens. Requests can be
added between any two steps and join the next one; finished requests free
their KV blocks immediately, so the batch never waits for its slowest member.

With a SafetyMonitor attached, every sampled position is also scored by the
safety probes from activations the same forward captured; a stop-mode
monitor ends a flagged sequence before the flagged token is emitted.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from safeinfer.engine.model_runner import ModelRunner, SeqStep
from safeinfer.engine.sampler import sample
from safeinfer.safety.monitor import SafetyMonitor, SafetySignal
from safeinfer.sampling_params import SamplingParams
from safeinfer.scheduler.scheduler import Scheduler, SchedulerConfig
from safeinfer.scheduler.sequence import Sequence


@dataclass(frozen=True)
class TokenOutput:
    seq_id: int
    token_id: int | None  # None only when a safety stop withheld this position's token
    finished: bool
    finish_reason: str | None  # "length", "stop", or "safety" once finished
    safety: SafetySignal | None = None  # set when a monitor is attached
    # [len(capture_layers), hidden] on CPU, the activation that predicted this position;
    # only with return_activations (offline data collection).
    activations: torch.Tensor | None = None


class LLMEngine:
    def __init__(
        self,
        runner: ModelRunner,
        config: SchedulerConfig,
        monitor: SafetyMonitor | None = None,
        return_activations: bool = False,
    ):
        if monitor and not set(monitor.layers) <= set(runner.capture_layers):
            raise ValueError(f"runner must capture layers {monitor.layers}, captures {runner.capture_layers}")
        if return_activations and not runner.capture_layers:
            raise ValueError("return_activations needs a runner with capture_layers")
        self.runner = runner
        self.scheduler = Scheduler(config, runner.block_manager, runner.model.config.eos_token_ids)
        self.monitor = monitor
        self.return_activations = return_activations
        self._next_seq_id = 0

    def add_request(self, prompt_token_ids: list[int], params: SamplingParams) -> int:
        """Queue a request; it joins the next step that has room. Returns its seq_id."""
        generator = None
        if params.seed is not None:
            generator = torch.Generator(device=self.runner.device)
            generator.manual_seed(params.seed)
        seq = Sequence(self._next_seq_id, list(prompt_token_ids), params, generator)
        self.scheduler.add(seq)
        self._next_seq_id += 1
        return seq.seq_id

    def has_unfinished_requests(self) -> bool:
        return self.scheduler.has_unfinished()

    @property
    def num_preemptions(self) -> int:
        return self.scheduler.num_preemptions

    def step(self) -> list[TokenOutput]:
        """Run one forward step; returns a token for each sequence that produced one."""
        if not self.scheduler.has_unfinished():
            return []
        scheduled = self.scheduler.schedule().scheduled
        if not scheduled:
            # The config checks guarantee the head of the queue fits an empty engine.
            raise RuntimeError("scheduler made no progress with requests pending")

        out = self.runner.step([SeqStep(s.seq.seq_id, s.token_ids) for s in scheduled])
        rows = [i for i, s in enumerate(scheduled) if s.samples]
        seqs = [scheduled[i].seq for i in rows]
        tokens = sample(out.logits[rows], [s.params for s in seqs], [s.generator for s in seqs])
        signals = self._observe(seqs, out.captured[rows] if self.monitor else None)
        stopped = set()
        if self.monitor and self.monitor.stop:
            stopped = {s.seq_id for s, sig in zip(seqs, signals, strict=True) if sig.flagged}

        acts = out.captured[rows].cpu() if self.return_activations else [None] * len(seqs)

        next_tokens = dict(zip(rows, tokens, strict=True))
        keep = [i for i, s in enumerate(scheduled) if s.seq.seq_id not in stopped]
        self.scheduler.update([scheduled[i] for i in keep], [next_tokens.get(i) for i in keep])
        for s in seqs:
            if s.seq_id in stopped:
                self.scheduler.finish(s, "safety")
            if self.monitor and s.finish_reason is not None:
                self.monitor.release(s.seq_id)

        return [
            TokenOutput(
                s.seq_id,
                None if s.seq_id in stopped else tok,
                s.finish_reason is not None,
                s.finish_reason,
                sig,
                act,
            )
            for s, tok, sig, act in zip(seqs, tokens, signals, acts, strict=True)
        ]

    def _observe(self, seqs: list[Sequence], captured: torch.Tensor | None) -> list[SafetySignal | None]:
        """Score each sampled sequence's current output position; Nones without a monitor."""
        if not self.monitor:
            return [None] * len(seqs)
        layers = self.runner.capture_layers
        acts = {layer: captured[:, layers.index(layer)] for layer in self.monitor.layers}
        # Positions are counted before this step's token is appended.
        return self.monitor.observe([s.seq_id for s in seqs], [len(s.output_token_ids) for s in seqs], acts)

    def generate(self, prompts: list[list[int]], params: list[SamplingParams]) -> list[list[int]]:
        """Offline convenience: run all prompts to completion, return output tokens in prompt order."""
        if len(prompts) != len(params):
            raise ValueError("prompts and params differ in length")
        if self.has_unfinished_requests():
            raise RuntimeError("generate() needs an idle engine; use add_request/step to mix with live requests")
        ids = [self.add_request(p, sp) for p, sp in zip(prompts, params, strict=True)]
        outputs: dict[int, list[int]] = {i: [] for i in ids}
        while self.has_unfinished_requests():
            for o in self.step():
                if o.token_id is not None:
                    outputs[o.seq_id].append(o.token_id)
        return [outputs[i] for i in ids]
