"""Continuous-batching engine: scheduler + model runner + sampler, one step at a time.

Each `step()` runs one forward over whatever the scheduler picked (new
prompts, prompt chunks, and decode tokens mixed), samples a token for every
sequence that reached its end, and returns those tokens. Requests can be
added between any two steps and join the next one; finished requests free
their KV blocks immediately, so the batch never waits for its slowest member.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from safeinfer.engine.model_runner import ModelRunner, SeqStep
from safeinfer.engine.sampler import sample
from safeinfer.sampling_params import SamplingParams
from safeinfer.scheduler.scheduler import Scheduler, SchedulerConfig
from safeinfer.scheduler.sequence import Sequence


@dataclass(frozen=True)
class TokenOutput:
    seq_id: int
    token_id: int
    finished: bool
    finish_reason: str | None  # "length" or "stop" once finished


class LLMEngine:
    def __init__(self, runner: ModelRunner, config: SchedulerConfig):
        self.runner = runner
        self.scheduler = Scheduler(config, runner.block_manager, runner.model.config.eos_token_ids)
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
        tokens = sample(
            out.logits[rows],
            [scheduled[i].seq.params for i in rows],
            [scheduled[i].seq.generator for i in rows],
        )
        next_tokens: list[int | None] = [None] * len(scheduled)
        for i, tok in zip(rows, tokens, strict=True):
            next_tokens[i] = tok
        self.scheduler.update(scheduled, next_tokens)

        return [
            TokenOutput(s.seq.seq_id, tok, s.seq.finish_reason is not None, s.seq.finish_reason)
            for s, tok in zip(scheduled, next_tokens, strict=True)
            if tok is not None
        ]

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
                outputs[o.seq_id].append(o.token_id)
        return [outputs[i] for i in ids]
