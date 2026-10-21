"""Paged attention in plain PyTorch: write new K/V into blocks, gather, run SDPA.

Gathering each sequence's context into a contiguous tensor copies the whole
KV history every step, so this is slow; it exists to be obviously correct.
The Triton kernel replaces the gather with in-place block reads and is
checked against this backend.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from safeinfer.attention.backend import AttentionMetadata
from safeinfer.memory.kv_cache import KVCache


class TorchPagedAttention:
    def __init__(self, kv_cache: KVCache):
        self.kv_cache = kv_cache

    def forward(
        self,
        layer_idx: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        meta: AttentionMetadata,
    ) -> torch.Tensor:
        self.kv_cache.write(layer_idx, k, v, meta.slot_mapping)

        out = torch.empty_like(q)
        start = 0
        for i, (q_len, ctx_len) in enumerate(zip(meta.query_lens, meta.context_lens, strict=True)):
            end = start + q_len
            ks, vs = self.kv_cache.gather(layer_idx, meta.block_tables[i], ctx_len)
            # [len, heads, dim] -> [1, heads, len, dim]
            qs, ks, vs = (t.transpose(0, 1).unsqueeze(0) for t in (q[start:end], ks, vs))
            o = F.scaled_dot_product_attention(
                qs, ks, vs, attn_mask=_causal_mask(q_len, ctx_len, q.device), enable_gqa=True
            )
            out[start:end] = o.squeeze(0).transpose(0, 1)
            start = end
        return out


def _causal_mask(q_len: int, ctx_len: int, device: torch.device) -> torch.Tensor | None:
    """Queries are the last q_len of ctx_len positions; each sees itself and everything before.

    SDPA's is_causal aligns the mask to the top-left, which is wrong once the
    query is shorter than the context, so build it explicitly.
    """
    if q_len == 1:
        if device.type == "mps":
            # torch 2.9's MPS single-query SDPA kernel (used when no mask is given) returned
            # NaN or wrong values nondeterministically on real Llama decodes (seen at ~1k context,
            # bf16). An explicit all-true mask routes to the general kernel, which is correct.
            return torch.ones(1, ctx_len, dtype=torch.bool, device=device)
        return None  # a single (newest) query sees the whole context
    q_pos = torch.arange(ctx_len - q_len, ctx_len, device=device)[:, None]
    k_pos = torch.arange(ctx_len, device=device)[None, :]
    return k_pos <= q_pos
