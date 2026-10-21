"""Preallocated physical KV blocks shared by all sequences."""

from __future__ import annotations

import torch

from safeinfer.config import ModelConfig


def bytes_per_block(cfg: ModelConfig, block_size: int, dtype: torch.dtype) -> int:
    """K and V for every layer, for one block of tokens."""
    elem = torch.empty((), dtype=dtype).element_size()
    return 2 * cfg.num_layers * block_size * cfg.num_kv_heads * cfg.head_dim * elem


def num_blocks_for_budget(cfg: ModelConfig, block_size: int, dtype: torch.dtype, budget_bytes: int) -> int:
    return budget_bytes // bytes_per_block(cfg, block_size, dtype)


def gpu_kv_budget(device: torch.device, memory_utilization: float, activation_reserve_bytes: int) -> int:
    """Bytes available for KV after the model is loaded.

    Call after loading weights: takes `memory_utilization` of total memory,
    minus what is already in use and a reserve for activations.
    """
    free, total = torch.cuda.mem_get_info(device)
    used = total - free
    return max(0, int(total * memory_utilization) - used - activation_reserve_bytes)


class KVCache:
    """k/v: [num_layers, num_blocks, block_size, num_kv_heads, head_dim]."""

    def __init__(
        self,
        cfg: ModelConfig,
        num_blocks: int,
        block_size: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ):
        shape = (cfg.num_layers, num_blocks, block_size, cfg.num_kv_heads, cfg.head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)
        self.num_blocks = num_blocks
        self.block_size = block_size

    def write(self, layer: int, k: torch.Tensor, v: torch.Tensor, slot_mapping: torch.Tensor) -> None:
        """Store new tokens' K/V ([num_tokens, kv_heads, head_dim]) at their slots."""
        self.k[layer].view(-1, *k.shape[1:])[slot_mapping] = k
        self.v[layer].view(-1, *v.shape[1:])[slot_mapping] = v

    def gather(self, layer: int, block_table: torch.Tensor, context_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """One sequence's first `context_len` tokens of K/V as contiguous tensors."""
        num_blocks = -(-context_len // self.block_size)
        blocks = block_table[:num_blocks]
        k = self.k[layer, blocks].flatten(0, 1)[:context_len]
        v = self.v[layer, blocks].flatten(0, 1)[:context_len]
        return k, v
