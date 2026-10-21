"""Benchmark entry point: one backend, one workload, one JSON result.

    python -m safeinfer.bench.run_bench --backend hf --model meta-llama/Llama-3.2-1B-Instruct \
        --dataset ShareGPT_V3_unfiltered_cleaned_split.json --num-requests 200 --rate 4

Backends: `hf` (HF generate, static batching), `engine-static` (our paged
engine, static batching), and `engine` (our paged engine, continuous
batching). Sweep --rate to trace the throughput / latency curve.
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
from safeinfer.bench.static_replay import replay_static_batching
from safeinfer.bench.workload import Request, build_workload, fixed_lengths, sharegpt_lengths

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def _run_hf(args, device: torch.device) -> tuple[list[RequestTiming], dict]:
    from transformers import AutoModelForCausalLM

    from safeinfer.bench.hf_baseline import hf_batch_runner

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=DTYPES[args.dtype], attn_implementation="sdpa")
    model = model.to(device).eval()
    requests = _make_requests(args, model.config.vocab_size)
    run_batch = hf_batch_runner(model)

    _warm_up(lambda: run_batch([_warm_up_request(requests[0])], time.perf_counter()), device)
    timings = replay_static_batching(requests, run_batch, args.max_batch_size)
    return timings, {"max_batch_size": args.max_batch_size}


def _make_runner(args, device: torch.device):
    from safeinfer.engine.model_runner import ModelRunner
    from safeinfer.memory.kv_cache import gpu_kv_budget, num_blocks_for_budget
    from safeinfer.model.loader import load_model

    dtype = DTYPES[args.dtype]
    model = load_model(args.model, dtype=dtype, device=device)
    num_blocks = args.num_blocks
    if num_blocks is None:
        if device.type != "cuda":
            raise SystemExit("--num-blocks is required off-GPU")
        budget = gpu_kv_budget(device, args.gpu_memory_utilization, activation_reserve_bytes=2**30)
        num_blocks = num_blocks_for_budget(model.config, args.block_size, dtype, budget)
    return ModelRunner(model, num_blocks=num_blocks, block_size=args.block_size)


def _run_engine_static(args, device: torch.device) -> tuple[list[RequestTiming], dict]:
    from safeinfer.bench.engine_static import engine_batch_runner, engine_can_admit

    runner = _make_runner(args, device)
    requests = _make_requests(args, runner.model.config.vocab_size)
    run_batch = engine_batch_runner(runner)

    _warm_up(lambda: run_batch([_warm_up_request(requests[0])], time.perf_counter()), device)
    timings = replay_static_batching(requests, run_batch, args.max_batch_size, engine_can_admit(runner))
    return timings, {
        "max_batch_size": args.max_batch_size,
        "num_blocks": runner.block_manager.num_blocks,
        "block_size": args.block_size,
    }


def _run_engine(args, device: torch.device) -> tuple[list[RequestTiming], dict]:
    from safeinfer.bench.continuous_replay import replay_continuous
    from safeinfer.engine.engine import LLMEngine
    from safeinfer.sampling_params import SamplingParams
    from safeinfer.scheduler.scheduler import SchedulerConfig

    runner = _make_runner(args, device)
    requests = _make_requests(args, runner.model.config.vocab_size)
    config = SchedulerConfig(
        max_num_seqs=args.max_batch_size,
        max_num_batched_tokens=args.max_num_batched_tokens or args.max_total_len,
        max_model_len=args.max_total_len,
        enable_chunked_prefill=args.enable_chunked_prefill,
    )
    engine = LLMEngine(runner, config)

    warm_params = SamplingParams(max_tokens=4, ignore_eos=True)
    _warm_up(lambda: engine.generate([requests[0].prompt_token_ids], [warm_params]), device)
    preemptions_before = engine.num_preemptions
    timings = replay_continuous(requests, engine)
    return timings, {
        **config.__dict__,
        "num_blocks": runner.block_manager.num_blocks,
        "block_size": args.block_size,
        "num_preemptions": engine.num_preemptions - preemptions_before,
    }


BACKENDS = {"hf": _run_hf, "engine-static": _run_engine_static, "engine": _run_engine}


def _warm_up_request(request: Request) -> Request:
    return Request(-1, 0.0, request.prompt_token_ids, 4)


def _warm_up(run, device: torch.device) -> None:
    """Compile kernels and grow the allocator outside the timed run."""
    run()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def _make_requests(args, vocab_size: int) -> list[Request]:
    rng = np.random.default_rng(args.seed)
    if args.dataset:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        lengths = sharegpt_lengths(args.dataset, tokenizer, args.num_requests, rng, args.max_total_len)
    else:
        lengths = fixed_lengths(args.num_requests, args.prompt_len, args.output_len)
    return build_workload(lengths, args.rate, vocab_size, seed=args.seed)


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
    p.add_argument(
        "--max-total-len", type=int, default=2048, help="drop dataset pairs longer than this; engine max_model_len"
    )
    p.add_argument("--num-requests", type=int, default=200)
    p.add_argument("--rate", type=float, default=float("inf"), help="requests/s (Poisson); inf = all at t=0")
    p.add_argument("--max-batch-size", type=int, default=32, help="static: batch size; engine: max running seqs")
    p.add_argument("--max-num-batched-tokens", type=int, help="engine: tokens per step (default --max-total-len)")
    p.add_argument("--enable-chunked-prefill", action="store_true", help="engine: split prompts across steps")
    p.add_argument("--block-size", type=int, default=16, help="engine: tokens per KV block")
    p.add_argument("--num-blocks", type=int, help="engine: KV blocks; default sizes from free GPU memory")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="engine: fraction of GPU memory to use")
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
