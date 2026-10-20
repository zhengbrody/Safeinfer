"""Benchmark entry point: one backend, one workload, one JSON result.

    python -m safeinfer.bench.run_bench --backend hf --model meta-llama/Llama-3.2-1B-Instruct \
        --dataset ShareGPT_V3_unfiltered_cleaned_split.json --num-requests 200 --rate 4

Sweep --rate to trace the throughput / latency curve. Backends for our own
engine and vLLM plug in here as they are built.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from safeinfer.bench.metrics import RequestTiming, summarize
from safeinfer.bench.workload import Request, build_workload, fixed_lengths, sharegpt_lengths

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def _run_hf(args, device: torch.device) -> tuple[list[RequestTiming], dict]:
    from transformers import AutoModelForCausalLM

    from safeinfer.bench.hf_baseline import run_static_batching

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=DTYPES[args.dtype], attn_implementation="sdpa")
    model = model.to(device).eval()
    requests = _make_requests(args, model.config.vocab_size)

    # Warm up kernels and the allocator outside the timed run.
    warmup = [Request(-1, 0.0, requests[0].prompt_token_ids, 4)]
    run_static_batching(model, warmup, max_batch_size=1)
    _reset_peak_memory(device)

    timings = run_static_batching(model, requests, max_batch_size=args.max_batch_size)
    return timings, {"max_batch_size": args.max_batch_size}


BACKENDS = {"hf": _run_hf}


def _make_requests(args, vocab_size: int) -> list[Request]:
    rng = np.random.default_rng(args.seed)
    if args.dataset:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        lengths = sharegpt_lengths(args.dataset, tokenizer, args.num_requests, rng, args.max_total_len)
    else:
        lengths = fixed_lengths(args.num_requests, args.prompt_len, args.output_len)
    return build_workload(lengths, args.rate, vocab_size, seed=args.seed)


def _reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def _environment(device: torch.device) -> dict:
    import transformers

    env = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
        "device": str(device),
    }
    if device.type == "cuda":
        env["gpu"] = torch.cuda.get_device_name(device)
    return env


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=sorted(BACKENDS), required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--dtype", choices=sorted(DTYPES), default="bfloat16")
    p.add_argument("--dataset", help="ShareGPT JSON; omit to use --prompt-len/--output-len for every request")
    p.add_argument("--prompt-len", type=int, default=512)
    p.add_argument("--output-len", type=int, default=128)
    p.add_argument("--max-total-len", type=int, default=2048, help="drop dataset pairs longer than this")
    p.add_argument("--num-requests", type=int, default=200)
    p.add_argument("--rate", type=float, default=float("inf"), help="requests/s (Poisson); inf = all at t=0")
    p.add_argument("--max-batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, help="write the result JSON here")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timings, backend_config = BACKENDS[args.backend](args, device)
    summary = summarize(timings)
    if device.type == "cuda":
        summary["peak_memory_gib"] = torch.cuda.max_memory_allocated(device) / 2**30

    result = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "backend_config": backend_config,
        "environment": _environment(device),
        "summary": summary,
        "requests": [t.__dict__ for t in timings],
    }
    for k, v in summary.items():
        print(f"{k:>26}: {v:.2f}" if isinstance(v, float) else f"{k:>26}: {v}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
