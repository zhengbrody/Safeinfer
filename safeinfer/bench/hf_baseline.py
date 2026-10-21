"""HuggingFace `generate` as a static-batching backend: the baseline we improve on.

Each batch is left-padded and every sequence runs for as many steps as the
longest one: together with static_replay's "new arrivals wait for the batch",
exactly the two inefficiencies paged KV and continuous batching remove.
Only each request's own `output_len` tokens count toward throughput.
"""

from __future__ import annotations

import time

import torch
from transformers import StoppingCriteria, StoppingCriteriaList

from safeinfer.bench.static_replay import RunBatch
from safeinfer.bench.workload import Request

PAD_TOKEN_ID = 0


class _StepClock(StoppingCriteria):
    """Records a timestamp after each generated token; never stops generation."""

    def __init__(self, t0: float, device: torch.device):
        self.t0 = t0
        self.device = device
        self.step_times: list[float] = []

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.step_times.append(time.perf_counter() - self.t0)
        return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)


def _left_pad(batch: list[Request], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(r.prompt_len for r in batch)
    input_ids = torch.full((len(batch), max_len), PAD_TOKEN_ID, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, r in enumerate(batch):
        input_ids[i, max_len - r.prompt_len :] = torch.tensor(r.prompt_token_ids)
        attention_mask[i, max_len - r.prompt_len :] = 1
    return input_ids.to(device), attention_mask.to(device)


def hf_batch_runner(model) -> RunBatch:
    """Wrap an HF causal LM as a static_replay `run_batch`."""

    def run_batch(batch: list[Request], t0: float) -> list[float]:
        input_ids, attention_mask = _left_pad(batch, model.device)
        steps = max(r.output_len for r in batch)
        clock = _StepClock(t0, model.device)
        model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=steps,
            min_new_tokens=steps,  # ignore EOS: output lengths come from the workload
            do_sample=False,
            pad_token_id=PAD_TOKEN_ID,
            stopping_criteria=StoppingCriteriaList([clock]),
        )
        return clock.step_times

    return run_batch
