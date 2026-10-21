"""Per-request generation settings."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SamplingParams:
    max_tokens: int = 16
    # 0 means greedy; top_k/top_p are then ignored.
    temperature: float = 0.0
    top_k: int = 0  # 0 disables
    top_p: float = 1.0
    # Seeds this request's own RNG, so its samples do not depend on what else
    # is in the batch, on batch order, or on preemption.
    seed: int | None = None
    # Generation stops after emitting any of these (the stop token is kept in the output).
    stop_token_ids: tuple[int, ...] = ()
    # Do not stop at the model's EOS tokens (benchmarks fix output lengths this way).
    ignore_eos: bool = False

    def __post_init__(self):
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if self.top_k < 0:
            raise ValueError("top_k must be >= 0")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")

    @property
    def greedy(self) -> bool:
        return self.temperature == 0
