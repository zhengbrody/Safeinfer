"""Benchmark workloads: Poisson arrivals with realistic prompt/output lengths.

Only the *lengths* of real conversations are used; prompt contents are random
token ids and every request is forced to generate exactly `output_len` tokens.
That keeps runs comparable across backends regardless of what the model says.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Keep random prompt tokens clear of the low/high ids where vocabularies put
# special and control tokens.
_TOKEN_MARGIN = 1000


@dataclass(frozen=True)
class Request:
    request_id: int
    arrival_time: float  # seconds after benchmark start
    prompt_token_ids: list[int]
    output_len: int

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_token_ids)


def poisson_arrival_times(n: int, rate: float, rng: np.random.Generator) -> np.ndarray:
    """Arrival times for a Poisson process; rate=inf sends everything at t=0 (offline mode)."""
    if math.isinf(rate):
        return np.zeros(n)
    return np.cumsum(rng.exponential(1.0 / rate, size=n))


def fixed_lengths(n: int, prompt_len: int, output_len: int) -> list[tuple[int, int]]:
    return [(prompt_len, output_len)] * n


def sharegpt_lengths(
    path: str | Path,
    tokenizer,
    n: int,
    rng: np.random.Generator,
    max_total_len: int = 2048,
) -> list[tuple[int, int]]:
    """(prompt_len, output_len) pairs from the first human/assistant turn of ShareGPT conversations.

    Filtering follows vLLM's benchmark: drop turns under 4 tokens and pairs
    whose total exceeds the context budget.
    """
    with open(path) as f:
        conversations = [c["conversations"] for c in json.load(f) if len(c.get("conversations", [])) >= 2]
    rng.shuffle(conversations)

    pairs: list[tuple[int, int]] = []
    for conv in conversations:
        prompt_len = len(tokenizer(conv[0]["value"]).input_ids)
        output_len = len(tokenizer(conv[1]["value"]).input_ids)
        if prompt_len < 4 or output_len < 4 or prompt_len + output_len > max_total_len:
            continue
        pairs.append((prompt_len, output_len))
        if len(pairs) == n:
            return pairs
    raise ValueError(f"only {len(pairs)} usable conversations in {path}, need {n}")


def build_workload(
    lengths: list[tuple[int, int]],
    rate: float,
    vocab_size: int,
    seed: int = 0,
) -> list[Request]:
    rng = np.random.default_rng(seed)
    arrivals = poisson_arrival_times(len(lengths), rate, rng)
    margin = min(_TOKEN_MARGIN, vocab_size // 4)
    return [
        Request(
            request_id=i,
            arrival_time=float(arrivals[i]),
            prompt_token_ids=rng.integers(margin, vocab_size - margin, size=p).tolist(),
            output_len=o,
        )
        for i, (p, o) in enumerate(lengths)
    ]
