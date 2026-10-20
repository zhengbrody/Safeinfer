"""Load HuggingFace safetensors checkpoints into LlamaForCausalLM."""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors import safe_open

from safeinfer.config import ModelConfig
from safeinfer.model.llama import LlamaForCausalLM, RotaryEmbedding


def resolve_model_dir(name_or_path: str | Path) -> Path:
    """A local directory as-is, otherwise download config + weights from the hub."""
    path = Path(name_or_path)
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(str(name_or_path), allow_patterns=["*.json", "*.safetensors"]))


def load_model(
    name_or_path: str | Path,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | str = "cuda",
) -> LlamaForCausalLM:
    model_dir = resolve_model_dir(name_or_path)
    cfg = ModelConfig.from_pretrained(model_dir)

    # Build on the meta device so no memory is spent on random init.
    with torch.device("meta"):
        model = LlamaForCausalLM(cfg)

    state = {}
    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors files in {model_dir}")
    for file in files:
        with safe_open(file, framework="pt", device=str(device)) as f:
            for key in f.keys():  # noqa: SIM118 -- safe_open handles are not iterable
                state[key] = f.get_tensor(key).to(dtype)

    if cfg.tie_word_embeddings:
        # Tied checkpoints usually omit lm_head; it shares embed_tokens' weight.
        state["lm_head.weight"] = state["model.embed_tokens.weight"]

    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    if cfg.tie_word_embeddings:
        # assign=True swapped in new Parameters; re-tie so both names share one tensor.
        model.lm_head.weight = model.model.embed_tokens.weight

    # RoPE inv_freq is a non-persistent buffer (not in the checkpoint), so it is
    # still on the meta device; rebuild it for real.
    model.model.rotary_emb = RotaryEmbedding(cfg.head_dim, cfg.rope).to(device)
    return model.eval()
