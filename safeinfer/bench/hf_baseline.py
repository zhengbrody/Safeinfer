"""HuggingFace `generate` with static batching: the baseline we improve on.

Simulates a server that, whenever it is idle, takes up to `max_batch_size`
requests that have already arrived and runs them to completion as one padded
batch. New arrivals wait until the whole batch finishes, and every sequence
runs for as many steps as the longest one: exactly the two inefficiencies
paged KV and continuous batching remove.

Timing is credited as if tokens were streamed: a request's first and last
tokens count when their decode steps finish, not when `generate` returns.
That flatters the baseline slightly, which keeps the comparison honest.
"""

from __future__ import annotations

import time
from collections import deque

import torch
from transformers import StoppingCriteria, StoppingCriteriaList

from safeinfer.bench.metrics import RequestTiming
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


def run_static_batching(model, requests: list[Request], max_batch_size: int) -> list[RequestTiming]:
    """Replay `requests` in real time against `model` (an HF causal LM)."""
    device = model.device
    pending = deque(sorted(requests, key=lambda r: r.arrival_time))
    timings: list[RequestTiming] = []
    t0 = time.perf_counter()

    while pending:
        wait = pending[0].arrival_time - (time.perf_counter() - t0)
        if wait > 0:
            time.sleep(wait)
        now = time.perf_counter() - t0
        batch = []
        while pending and len(batch) < max_batch_size and pending[0].arrival_time <= now:
            batch.append(pending.popleft())

        input_ids, attention_mask = _left_pad(batch, device)
        steps = max(r.output_len for r in batch)
        clock = _StepClock(t0, device)
        model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=steps,
            min_new_tokens=steps,  # ignore EOS: output lengths come from the workload
            do_sample=False,
            pad_token_id=PAD_TOKEN_ID,
            stopping_criteria=StoppingCriteriaList([clock]),
        )
        for r in batch:
            timings.append(
                RequestTiming(
                    request_id=r.request_id,
                    prompt_len=r.prompt_len,
                    output_len=r.output_len,
                    arrival_time=r.arrival_time,
                    first_token_time=clock.step_times[0],
                    finish_time=clock.step_times[r.output_len - 1],
                )
            )
    return timings
