"""Real-time replay of a workload against a static-batching server.

Whenever the server is idle it takes requests that have already arrived, up
to `max_batch_size` and while `can_admit` allows, and runs them to completion
as one batch. New arrivals wait for the whole batch. HF `generate` and our
engine's static mode both plug in as a `run_batch` function, so the
comparison differs only in what runs a batch.

Timing is credited as if tokens were streamed: a request's first and last
tokens count when their decode steps finish, not when the batch returns.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable

from safeinfer.bench.metrics import RequestTiming
from safeinfer.bench.workload import Request

# Runs one batch to completion; returns a timestamp (seconds since t0) per decode step.
RunBatch = Callable[[list[Request], float], list[float]]
CanAdmit = Callable[[list[Request], Request], bool]


def replay_static_batching(
    requests: list[Request],
    run_batch: RunBatch,
    max_batch_size: int,
    can_admit: CanAdmit | None = None,
) -> list[RequestTiming]:
    pending = deque(sorted(requests, key=lambda r: r.arrival_time))
    timings: list[RequestTiming] = []
    t0 = time.perf_counter()

    while pending:
        wait = pending[0].arrival_time - (time.perf_counter() - t0)
        if wait > 0:
            time.sleep(wait)
        now = time.perf_counter() - t0
        batch: list[Request] = []
        while pending and len(batch) < max_batch_size and pending[0].arrival_time <= now:
            if can_admit and not can_admit(batch, pending[0]):
                break
            batch.append(pending.popleft())
        if not batch:
            raise RuntimeError(f"request {pending[0].request_id} can never be admitted (too large for the server)")

        step_times = run_batch(batch, t0)
        for r in batch:
            timings.append(
                RequestTiming(
                    request_id=r.request_id,
                    prompt_len=r.prompt_len,
                    output_len=r.output_len,
                    arrival_time=r.arrival_time,
                    first_token_time=step_times[0],
                    finish_time=step_times[r.output_len - 1],
                )
            )
    return timings
