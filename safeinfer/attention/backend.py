"""Attention backend interface.

The model always runs on a *flattened* token batch: every sequence's tokens
are concatenated into one [num_tokens, ...] tensor, and AttentionMetadata says
where each sequence starts and ends. This is the layout continuous batching
needs (prefill and decode sequences of different lengths in one step), so the
model code never changes when we swap backends — only the backend does.

Backends:
    DenseAttention  (dense.py)        causal attention over the given tokens only,
                                       no KV cache; the correctness reference.
    paged backends  (week 2 onward)   read/write a block-paged KV cache.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass
class AttentionMetadata:
    # Number of query tokens contributed by each sequence, in batch order.
    # sum(query_lens) == num_tokens of the flattened batch.
    query_lens: list[int]

    @classmethod
    def for_prefill(cls, prompt_lens: list[int]) -> AttentionMetadata:
        return cls(query_lens=list(prompt_lens))

    @property
    def num_tokens(self) -> int:
        return sum(self.query_lens)

    def last_token_indices(self, device: torch.device | str) -> torch.Tensor:
        """Index of each sequence's final token in the flattened batch."""
        ends = torch.tensor(self.query_lens, device=device).cumsum(0)
        return ends - 1


class AttentionBackend(Protocol):
    def forward(
        self,
        layer_idx: int,
        q: torch.Tensor,  # [num_tokens, num_heads, head_dim]
        k: torch.Tensor,  # [num_tokens, num_kv_heads, head_dim]
        v: torch.Tensor,  # [num_tokens, num_kv_heads, head_dim]
        meta: AttentionMetadata,
    ) -> torch.Tensor:  # [num_tokens, num_heads, head_dim]
        ...
