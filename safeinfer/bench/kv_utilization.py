"""KV memory: contiguous max-length preallocation vs paged blocks, on real lengths.

Pure arithmetic over a workload's (prompt_len, output_len) pairs, so it runs
anywhere. For a fixed KV budget it reports, per scheme:

    utilization   slots holding a token / slots reserved, averaged over each
                  request's lifetime (its length grows from prompt to prompt+output)
    concurrency   how many requests the budget holds at once when each must
                  reserve its worst case up front (no preemption)

Contiguous reserves `max_model_len` per request (HF-style static cache).
Paged reserves ceil(len / block_size) blocks, growing as it decodes; its
concurrency reserves for the final length since static mode cannot preempt.

    python -m safeinfer.bench.kv_utilization --dataset data/sharegpt.json \
        --model meta-llama/Llama-3.2-1B-Instruct --kv-budget-gib 16
"""

from __future__ import annotations

import argparse
import json

import numpy as np


def _ceil_to(x: np.ndarray, m: int) -> np.ndarray:
    return -(-x // m) * m


def lifetime_utilization(prompt: np.ndarray, output: np.ndarray, reserved_at) -> float:
    """Stored / reserved slot-steps, summed over every decode step of every request."""
    stored = reserved = 0
    for p, o in zip(prompt, output, strict=True):
        lengths = np.arange(p + 1, p + o + 1)  # length after each generated token
        stored += lengths.sum()
        reserved += reserved_at(lengths).sum()
    return stored / reserved


def analyze(
    lengths: list[tuple[int, int]],
    max_model_len: int,
    block_sizes: list[int],
    budget_tokens: int,
) -> list[dict]:
    prompt = np.array([p for p, _ in lengths])
    output = np.array([o for _, o in lengths])
    final = prompt + output
    rows = [
        {
            "scheme": f"contiguous (max_len={max_model_len})",
            "utilization": lifetime_utilization(prompt, output, lambda n: np.full_like(n, max_model_len)),
            "concurrency": budget_tokens // max_model_len,
        }
    ]
    for bs in block_sizes:
        rows.append(
            {
                "scheme": f"paged (block={bs})",
                "utilization": lifetime_utilization(prompt, output, lambda n, bs=bs: _ceil_to(n, bs)),
                # Average requests that fit when each reserves its final length in blocks.
                "concurrency": int(budget_tokens // _ceil_to(final, bs).mean()),
            }
        )
    return rows


def main() -> None:
    from transformers import AutoTokenizer

    from safeinfer.bench.workload import sharegpt_lengths
    from safeinfer.config import ModelConfig
    from safeinfer.memory.kv_cache import bytes_per_block
    from safeinfer.model.loader import resolve_model_dir

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--model", required=True, help="for the tokenizer and per-token KV size")
    p.add_argument("--num-requests", type=int, default=1000)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--kv-budget-gib", type=float, default=16.0)
    p.add_argument("--block-sizes", type=int, nargs="+", default=[8, 16, 32])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", help="write rows as JSON here")
    args = p.parse_args()

    import torch

    model_dir = resolve_model_dir(args.model)
    cfg = ModelConfig.from_pretrained(model_dir)
    bytes_per_token = bytes_per_block(cfg, 1, torch.bfloat16)
    budget_tokens = int(args.kv_budget_gib * 2**30) // bytes_per_token

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    rng = np.random.default_rng(args.seed)
    lengths = sharegpt_lengths(args.dataset, tokenizer, args.num_requests, rng, args.max_model_len)
    final = np.array([a + b for a, b in lengths])

    print(f"{len(lengths)} requests, final length mean {final.mean():.0f} / p50 {np.median(final):.0f} tokens")
    print(
        f"KV budget {args.kv_budget_gib} GiB = {budget_tokens:,} tokens ({bytes_per_token // 1024} KiB/token, bf16)\n"
    )
    rows = analyze(lengths, args.max_model_len, args.block_sizes, budget_tokens)
    print(f"| {'scheme':<26} | utilization | waste | max concurrent |")
    print(f"|{'-' * 28}|-------------|-------|----------------|")
    for r in rows:
        u = r["utilization"]
        print(f"| {r['scheme']:<26} | {u:>10.1%} | {1 - u:>5.1%} | {r['concurrency']:>14,} |")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"args": vars(args), "budget_tokens": budget_tokens, "rows": rows}, f, indent=2)


if __name__ == "__main__":
    main()
