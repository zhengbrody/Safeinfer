"""A request's state as it moves through the scheduler."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from safeinfer.sampling_params import SamplingParams

if TYPE_CHECKING:
    import torch


class SequenceStatus(enum.Enum):
    WAITING = "waiting"  # holds no KV blocks
    RUNNING = "running"  # holds KV for its first num_computed_tokens tokens
    FINISHED = "finished"


@dataclass(eq=False)
class Sequence:
    seq_id: int
    prompt_token_ids: list[int]
    params: SamplingParams
    generator: torch.Generator | None = None  # per-request RNG; None for unseeded
    output_token_ids: list[int] = field(default_factory=list)
    status: SequenceStatus = SequenceStatus.WAITING
    # Tokens whose K/V are in the cache. Reset to 0 on preemption (recompute).
    num_computed_tokens: int = 0
    finish_reason: str | None = None  # "length" or "stop"

    @property
    def num_tokens(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    @property
    def num_uncomputed_tokens(self) -> int:
        return self.num_tokens - self.num_computed_tokens

    def token_ids(self, start: int, end: int) -> list[int]:
        """Tokens [start, end) of prompt + output, without copying the whole sequence."""
        p = len(self.prompt_token_ids)
        if end <= p:
            return self.prompt_token_ids[start:end]
        if start >= p:
            return self.output_token_ids[start - p : end - p]
        return self.prompt_token_ids[start:] + self.output_token_ids[: end - p]
