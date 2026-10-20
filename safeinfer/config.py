"""Model configuration parsed directly from a HuggingFace config.json.

We read the raw JSON instead of going through transformers so the engine has
no runtime dependency on it (transformers is only used as a test oracle and
for the HF baseline).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED_MODEL_TYPES = ("llama", "qwen2")


@dataclass(frozen=True)
class RopeConfig:
    theta: float
    rope_type: str = "default"
    # llama3-style frequency scaling; ignored when rope_type == "default".
    factor: float = 1.0
    low_freq_factor: float = 1.0
    high_freq_factor: float = 4.0
    original_max_position_embeddings: int = 8192


@dataclass(frozen=True)
class ModelConfig:
    model_type: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rms_norm_eps: float
    max_position_embeddings: int
    rope: RopeConfig
    tie_word_embeddings: bool
    attention_bias: bool
    eos_token_ids: tuple[int, ...] = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, cfg: dict) -> ModelConfig:
        model_type = cfg["model_type"]
        if model_type not in SUPPORTED_MODEL_TYPES:
            raise ValueError(f"unsupported model_type {model_type!r}; expected one of {SUPPORTED_MODEL_TYPES}")
        if cfg.get("hidden_act", "silu") != "silu":
            raise ValueError(f"unsupported hidden_act {cfg['hidden_act']!r}")

        num_heads = cfg["num_attention_heads"]
        eos = cfg.get("eos_token_id")
        eos_ids = tuple(eos) if isinstance(eos, list) else (() if eos is None else (eos,))

        return cls(
            model_type=model_type,
            vocab_size=cfg["vocab_size"],
            hidden_size=cfg["hidden_size"],
            intermediate_size=cfg["intermediate_size"],
            num_layers=cfg["num_hidden_layers"],
            num_heads=num_heads,
            num_kv_heads=cfg.get("num_key_value_heads", num_heads),
            head_dim=cfg.get("head_dim") or cfg["hidden_size"] // num_heads,
            rms_norm_eps=cfg["rms_norm_eps"],
            max_position_embeddings=cfg["max_position_embeddings"],
            rope=_parse_rope(cfg),
            tie_word_embeddings=cfg.get("tie_word_embeddings", False),
            # Qwen2 always has q/k/v biases; Llama exposes it as a flag.
            attention_bias=model_type == "qwen2" or cfg.get("attention_bias", False),
            eos_token_ids=eos_ids,
        )

    @classmethod
    def from_pretrained(cls, model_dir: str | Path) -> ModelConfig:
        with open(Path(model_dir) / "config.json") as f:
            return cls.from_dict(json.load(f))


def _parse_rope(cfg: dict) -> RopeConfig:
    # transformers >= 5 writes `rope_parameters` (theta inside); older hub
    # checkpoints use top-level `rope_theta` plus an optional `rope_scaling`.
    params = cfg.get("rope_parameters") or cfg.get("rope_scaling") or {}
    theta = params.get("rope_theta", cfg.get("rope_theta", 10000.0))
    rope_type = params.get("rope_type", params.get("type", "default"))
    if rope_type not in ("default", "llama3"):
        raise ValueError(f"unsupported rope_type {rope_type!r}")
    if rope_type == "default":
        return RopeConfig(theta=theta)
    return RopeConfig(
        theta=theta,
        rope_type=rope_type,
        factor=params["factor"],
        low_freq_factor=params["low_freq_factor"],
        high_freq_factor=params["high_freq_factor"],
        original_max_position_embeddings=params["original_max_position_embeddings"],
    )
