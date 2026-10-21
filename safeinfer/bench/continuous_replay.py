"""Real-time replay of a workload against the continuous-batching engine.

Arrived requests are added before every step, so they join the batch at the
next iteration instead of waiting for the current batch to drain (the
static_replay loop). Token timestamps are taken after each step returns,
which is after the sampled tokens reached the host.
"""

from __future__ import annotations

import time
from collections import deque

from safeinfer.bench.metrics import RequestTiming
from safeinfer.bench.workload import Request
from safeinfer.engine.engine import LLMEngine
from safeinfer.sampling_params import SamplingParams


def replay_continuous(requests: list[Request], engine: LLMEngine) -> list[RequestTiming]:
    pending = deque(sorted(requests, key=lambda r: r.arrival_time))
    by_seq: dict[int, Request] = {}
    first_token: dict[int, float] = {}
    timings: list[RequestTiming] = []
    t0 = time.perf_counter()

    while pending or engine.has_unfinished_requests():
        if not engine.has_unfinished_requests():
            wait = pending[0].arrival_time - (time.perf_counter() - t0)
            if wait > 0:
                time.sleep(wait)
        now = time.perf_counter() - t0
        while pending and pending[0].arrival_time <= now:
            r = pending.popleft()
            # Greedy and EOS ignored: every request generates exactly its workload output_len.
            seq_id = engine.add_request(r.prompt_token_ids, SamplingParams(max_tokens=r.output_len, ignore_eos=True))
            by_seq[seq_id] = r

        outputs = engine.step()
        t = time.perf_counter() - t0
        for o in outputs:
            first_token.setdefault(o.seq_id, t)
            if o.finished:
                r = by_seq.pop(o.seq_id)
                timings.append(
                    RequestTiming(
                        request_id=r.request_id,
                        prompt_len=r.prompt_len,
                        output_len=r.output_len,
                        arrival_time=r.arrival_time,
                        first_token_time=first_token.pop(o.seq_id),
                        finish_time=t,
                    )
                )
    return timings
