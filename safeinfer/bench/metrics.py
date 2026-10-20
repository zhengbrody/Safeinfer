"""Per-request timings and the serving metrics derived from them.

All timestamps are seconds on the same clock as Request.arrival_time
(time since benchmark start).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RequestTiming:
    request_id: int
    prompt_len: int
    output_len: int
    arrival_time: float
    first_token_time: float  # when the first output token was produced
    finish_time: float  # when the last requested output token was produced

    @property
    def ttft(self) -> float:
        return self.first_token_time - self.arrival_time

    @property
    def tpot(self) -> float | None:
        """Mean time per output token after the first; undefined for 1-token outputs."""
        if self.output_len < 2:
            return None
        return (self.finish_time - self.first_token_time) / (self.output_len - 1)

    @property
    def e2e_latency(self) -> float:
        return self.finish_time - self.arrival_time


def summarize(timings: list[RequestTiming]) -> dict[str, float]:
    """Throughput and latency percentiles; latencies in milliseconds."""
    if not timings:
        raise ValueError("no timings to summarize")
    start = min(t.arrival_time for t in timings)
    duration = max(t.finish_time for t in timings) - start
    ttft = np.array([t.ttft for t in timings]) * 1e3
    tpot = np.array([t.tpot for t in timings if t.tpot is not None]) * 1e3
    e2e = np.array([t.e2e_latency for t in timings]) * 1e3

    summary = {
        "num_requests": len(timings),
        "duration_s": duration,
        "request_throughput": len(timings) / duration,
        "output_token_throughput": sum(t.output_len for t in timings) / duration,
        "total_token_throughput": sum(t.prompt_len + t.output_len for t in timings) / duration,
    }
    for name, values in (("ttft_ms", ttft), ("tpot_ms", tpot), ("e2e_ms", e2e)):
        if values.size:
            summary[f"mean_{name}"] = float(values.mean())
            summary[f"p50_{name}"] = float(np.percentile(values, 50))
            summary[f"p99_{name}"] = float(np.percentile(values, 99))
    return summary
