"""Cache-free causal attention: the correctness reference for every other backend."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from safeinfer.attention.backend import AttentionMetadata


class DenseAttention:
    """Full causal self-attention within each sequence; nothing is cached.

    Each sequence's query tokens are treated as its whole context, so this is
    only valid for prefill-from-scratch. It runs one SDPA call per sequence,
    which is slow but obviously correct.
    """

    def forward(
        self,
        layer_idx: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        meta: AttentionMetadata,
    ) -> torch.Tensor:
        out = torch.empty_like(q)
        start = 0
        for n in meta.query_lens:
            end = start + n
            # [n, heads, dim] -> [1, heads, n, dim]
            qs, ks, vs = (t[start:end].transpose(0, 1).unsqueeze(0) for t in (q, k, v))
            o = F.scaled_dot_product_attention(qs, ks, vs, is_causal=True, enable_gqa=True)
            out[start:end] = o.squeeze(0).transpose(0, 1)
            start = end
        return out
