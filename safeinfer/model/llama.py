"""Hand-written Llama / Qwen2 forward pass over a flattened token batch.

Module and parameter names mirror HuggingFace exactly so checkpoints load with
a plain state_dict, and numerics follow HF's op order (fp32 RMSNorm and RoPE)
so logits can be compared against HF directly.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from safeinfer.attention.backend import AttentionBackend, AttentionMetadata
from safeinfer.config import ModelConfig, RopeConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, rope: RopeConfig):
        super().__init__()
        inv_freq = 1.0 / (rope.theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim))
        if rope.rope_type == "llama3":
            inv_freq = _llama3_scale(inv_freq, rope)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, positions: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """cos/sin of shape [num_tokens, head_dim], computed in fp32 like HF."""
        freqs = positions.float()[:, None] * self.inv_freq[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


def _llama3_scale(inv_freq: torch.Tensor, rope: RopeConfig) -> torch.Tensor:
    """Llama 3.1+ frequency-dependent RoPE scaling (long-context extension)."""
    old_ctx = rope.original_max_position_embeddings
    low_freq_wavelen = old_ctx / rope.low_freq_factor
    high_freq_wavelen = old_ctx / rope.high_freq_factor
    wavelen = 2 * math.pi / inv_freq

    scaled = torch.where(wavelen > low_freq_wavelen, inv_freq / rope.factor, inv_freq)
    smooth = (old_ctx / wavelen - rope.low_freq_factor) / (rope.high_freq_factor - rope.low_freq_factor)
    smoothed = (1 - smooth) * scaled / rope.factor + smooth * scaled
    is_medium = (wavelen >= high_freq_wavelen) & (wavelen <= low_freq_wavelen)
    return torch.where(is_medium, smoothed, scaled)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [num_tokens, heads, head_dim]; cos/sin broadcast over heads.
    cos, sin = cos[:, None, :], sin[:, None, :]
    return x * cos + _rotate_half(x) * sin


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = cfg.num_heads
        self.num_kv_heads = cfg.num_kv_heads
        self.head_dim = cfg.head_dim
        bias = cfg.attention_bias
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.num_heads * cfg.head_dim, bias=bias)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim, bias=bias)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim, bias=bias)
        self.o_proj = nn.Linear(cfg.num_heads * cfg.head_dim, cfg.hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        backend: AttentionBackend,
        meta: AttentionMetadata,
    ) -> torch.Tensor:
        n = x.shape[0]
        q = self.q_proj(x).view(n, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(n, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(n, self.num_kv_heads, self.head_dim)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        out = backend.forward(self.layer_idx, q, k, v, meta)
        return self.o_proj(out.reshape(n, -1))


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(cfg, layer_idx)
        self.mlp = MLP(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, x, cos, sin, backend, meta) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, backend, meta)
        return x + self.mlp(self.post_attention_layernorm(x))


class LlamaBackbone(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(DecoderLayer(cfg, i) for i in range(cfg.num_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(cfg.head_dim, cfg.rope)


class LlamaForCausalLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.config = cfg
        self.model = LlamaBackbone(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,  # [num_tokens]
        positions: torch.Tensor,  # [num_tokens]
        backend: AttentionBackend,
        meta: AttentionMetadata,
        capture_layer: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Returns (final normed hidden states, residual stream after `capture_layer`).

        The captured activation is what the safety probe reads; it comes from
        the same forward pass, so probing never costs an extra forward. It
        equals HF's `hidden_states[capture_layer + 1]`.
        """
        x = self.model.embed_tokens(input_ids)
        cos, sin = self.model.rotary_emb(positions, x.dtype)
        captured = None
        for i, layer in enumerate(self.model.layers):
            x = layer(x, cos, sin, backend, meta)
            if i == capture_layer:
                captured = x
        return self.model.norm(x), captured

    @torch.inference_mode()
    def compute_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Project to vocab; callers pass only the rows they will sample from."""
        return self.lm_head(hidden).float()
