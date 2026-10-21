"""Attention backend interface.

The model always runs on a *flattened* token batch: every sequence's new
tokens are concatenated into one [num_tokens, ...] tensor, and
AttentionMetadata says where each sequence starts and how much context it
has. A sequence's new tokens are the last `query_len` of its `context_len`
tokens, which covers every case with one description:

    prefill          query_len == context_len
    decode           query_len == 1
    chunked prefill  1 < query_len < context_len

So the model code never changes when we swap backends — only the backend does.

Backends:
    DenseAttention       (dense.py)        no cache; prefill from scratch only.
                                            The correctness reference.
    TorchPagedAttention  (torch_paged.py)  block-paged KV cache, gather + SDPA.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass
class AttentionMetadata:
    # New tokens per sequence, in batch order; sum == num_tokens.
    query_lens: list[int]
    # Tokens of context per sequence *including* this step's new tokens.
    context_lens: list[int]
    # Cache slot for each new token's K/V ([num_tokens]); None for cache-free backends.
    slot_mapping: torch.Tensor | None = None
    # Physical block ids per sequence ([num_seqs, max_blocks], right-padded).
    block_tables: torch.Tensor | None = None

    @classmethod
    def for_prefill(cls, prompt_lens: list[int]) -> AttentionMetadata:
        """Cache-free prefill: every sequence's context is exactly its new tokens."""
        return cls(query_lens=list(prompt_lens), context_lens=list(prompt_lens))

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
