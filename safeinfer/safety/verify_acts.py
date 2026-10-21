"""Checks stored activations against an independent CPU fp32 recomputation.

Offline QA for collect.py. For a random sample of generations, re-runs
prompt + generated tokens through the model once (teacher forcing, cache-free
dense attention, fp32 on CPU) and compares every stored position's
activation by cosine similarity. bf16 drift stays well above 0.99; a kernel
bug (the MPS decode-SDPA one this caught) shows up as NaN or collapses it.

    python -m safeinfer.safety.verify_acts --gens data/safety/gens_v2 --prompts data/safety/prompts.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from safeinfer.attention.backend import AttentionMetadata
from safeinfer.attention.dense import DenseAttention
from safeinfer.model.loader import load_model, resolve_model_dir
from safeinfer.safety.collect import CHAT_DATE


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gens", type=Path, required=True)
    p.add_argument("--prompts", type=Path, required=True)
    p.add_argument("--num", type=int, default=60)
    p.add_argument("--min-cosine", type=float, default=0.98)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from transformers import AutoTokenizer

    meta = json.loads((args.gens / "meta.json").read_text())
    prompts = {r["id"]: r for r in map(json.loads, args.prompts.read_text().splitlines())}
    gens = [json.loads(line) for line in (args.gens / "gens.jsonl").read_text().splitlines()]
    # Oversample long contexts, where the decode path differs most from prefill.
    gens.sort(key=lambda g: g["prompt_len"] + len(g["output_token_ids"]))
    rng = np.random.default_rng(args.seed)
    half = args.num // 2
    longest = range(len(gens) - (args.num - half), len(gens))
    sample = list(rng.choice(len(gens), size=half, replace=False)) + list(longest)

    model_dir = resolve_model_dir(meta["model"])
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = load_model(model_dir, dtype=torch.float32, device="cpu")
    layers = tuple(meta["position_layers"])
    acts = {layer: np.load(args.gens / f"acts_L{layer}.npy", mmap_mode="r") for layer in layers}

    worst = []
    for k in sample:
        g = gens[k]
        ids = list(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompts[g["id"]]["prompt"]}],
                add_generation_prompt=True,
                return_dict=False,
                date_string=CHAT_DATE,
            )
        )
        assert len(ids) == g["prompt_len"], "chat template changed since collection"
        seq = ids + g["output_token_ids"][:-1]  # position t is predicted from token prompt_len - 1 + t
        meta_ = AttentionMetadata.for_prefill([len(seq)])
        _, cap = model(torch.tensor(seq), torch.arange(len(seq)), DenseAttention(), meta_, capture_layers=layers)
        ref = cap[len(ids) - 1 :]  # [n, layers, hidden]
        n = len(g["output_token_ids"])
        for j, layer in enumerate(layers):
            stored = torch.from_numpy(np.asarray(acts[layer][g["id"], :n], dtype=np.float32))
            cos = torch.nn.functional.cosine_similarity(stored, ref[:, j], dim=-1).nan_to_num(-1.0)
            worst.append((cos.min().item(), g["id"], layer, int(cos.argmin()), g["prompt_len"] + n))

    worst.sort()
    bad = [w for w in worst if w[0] < args.min_cosine]
    print(f"checked {len(sample)} generations x {len(layers)} layers; min cosine {worst[0][0]:.4f}")
    for c, gid, layer, pos, total in worst[:5]:
        print(f"  cos {c:.4f}  prompt {gid}  layer {layer}  position {pos}  total len {total}")
    if bad:
        raise SystemExit(f"{len(bad)} (generation, layer) pairs below cosine {args.min_cosine}")
    print("OK")


if __name__ == "__main__":
    main()
