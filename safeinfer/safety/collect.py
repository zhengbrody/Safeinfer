"""Generates a response for every prompt with our engine and stores the activations probes train on.

Offline tooling. Output directory:

    gens.jsonl        one line per finished prompt: id, response tokens/text, finish reason
    acts_L{l}.npy     [num_prompts, max_tokens, hidden] fp16, per output position, for --position-layers
    pooled.npy        [num_prompts, 2, num_layers, hidden] fp16: position 0 (prompt end) and the
                      mean over output positions, for every layer (cheap layer sweep)
    meta.json         model, dtype, layers, max_tokens, chat-template date

Position t holds the activation that predicted output token t (position 0 =
the prompt's last token), exactly what the SafetyMonitor scores online.
Rows are indexed by prompt id; unwritten positions are zero, and gens.jsonl
says how many are valid. Rerunning resumes: prompts already in gens.jsonl
are skipped.

    python -m safeinfer.safety.collect --prompts data/safety/prompts.jsonl --out data/safety/gens \
        --model meta-llama/Llama-3.2-1B-Instruct --device mps
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from safeinfer.engine.engine import LLMEngine
from safeinfer.engine.model_runner import ModelRunner
from safeinfer.memory.kv_cache import num_blocks_for_budget
from safeinfer.model.loader import load_model, resolve_model_dir
from safeinfer.sampling_params import SamplingParams
from safeinfer.scheduler.scheduler import SchedulerConfig

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
# Llama 3.x chat templates stamp today's date into the system prompt; pin it so reruns match.
CHAT_DATE = "26 Jul 2024"


def _open_memmap(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    mode = "r+" if path.exists() else "w+"
    arr = np.lib.format.open_memmap(path, mode=mode, dtype=np.float16, shape=shape)
    if arr.shape != shape:
        raise ValueError(f"{path} has shape {arr.shape}, expected {shape}; use a fresh --out")
    return arr


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prompts", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", choices=sorted(DTYPES), default="bfloat16")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--position-layers", type=int, nargs="+", default=[5, 8, 11])
    p.add_argument("--max-num-seqs", type=int, default=32)
    p.add_argument("--kv-gib", type=float, default=2.0)
    p.add_argument("--limit", type=int, help="only the first N prompts (smoke runs)")
    args = p.parse_args()

    from transformers import AutoTokenizer

    records = [json.loads(line) for line in args.prompts.read_text().splitlines()]
    if args.limit:
        records = records[: args.limit]
    model_dir = resolve_model_dir(args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    dtype = DTYPES[args.dtype]
    model = load_model(model_dir, dtype=dtype, device=args.device)
    cfg = model.config
    layers = tuple(range(cfg.num_layers))
    if not set(args.position_layers) <= set(layers):
        raise SystemExit(f"--position-layers must be in 0..{cfg.num_layers - 1}")

    args.out.mkdir(parents=True, exist_ok=True)
    meta = {
        "model": args.model,
        "dtype": args.dtype,
        "device": args.device,
        "max_tokens": args.max_tokens,
        "position_layers": args.position_layers,
        "num_layers": cfg.num_layers,
        "chat_date": CHAT_DATE,
        "prompts": str(args.prompts),
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))
    n = max(r["id"] for r in records) + 1
    per_pos = {
        layer: _open_memmap(args.out / f"acts_L{layer}.npy", (n, args.max_tokens, cfg.hidden_size))
        for layer in args.position_layers
    }
    pooled = _open_memmap(args.out / "pooled.npy", (n, 2, cfg.num_layers, cfg.hidden_size))

    gens_path = args.out / "gens.jsonl"
    done = set()
    if gens_path.exists():
        done = {json.loads(line)["id"] for line in gens_path.read_text().splitlines()}

    block_size = 16
    budget = int(args.kv_gib * 2**30)
    runner = ModelRunner(
        model, num_blocks_for_budget(cfg, block_size, dtype, budget), block_size, capture_layers=layers
    )
    config = SchedulerConfig(
        max_num_seqs=args.max_num_seqs, max_num_batched_tokens=4096, max_model_len=args.max_model_len
    )
    engine = LLMEngine(runner, config, return_activations=True)
    params = SamplingParams(max_tokens=args.max_tokens)  # greedy; stops at EOS

    seq_to_record: dict[int, dict] = {}
    skipped = 0
    for r in records:
        if r["id"] in done:
            continue
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": r["prompt"]}],
            add_generation_prompt=True,
            return_dict=False,
            date_string=CHAT_DATE,
        )
        if len(ids) + args.max_tokens > args.max_model_len:
            skipped += 1
            continue
        seq_to_record[engine.add_request(list(ids), params)] = {"id": r["id"], "prompt_len": len(ids)}
    print(f"{len(seq_to_record)} to generate, {len(done)} already done, {skipped} too long")

    tokens: dict[int, list[int]] = {s: [] for s in seq_to_record}
    pos0: dict[int, torch.Tensor] = {}
    sums: dict[int, torch.Tensor] = {}
    position_index = [layers.index(layer) for layer in args.position_layers]
    t0, finished = time.perf_counter(), 0
    with open(gens_path, "a") as out:
        steps = 0
        while engine.has_unfinished_requests():
            steps += 1
            if args.device == "mps" and steps % 200 == 0:
                # Every step has a new batch shape; MPS caches freed buffers per shape and runs out.
                torch.mps.empty_cache()
            for o in engine.step():
                rid = seq_to_record[o.seq_id]["id"]
                t = len(tokens[o.seq_id])
                act = o.activations.float()  # [num_layers, hidden]
                if not torch.isfinite(act).all():
                    # Seen on MPS under memory pressure: fail loudly rather than store poisoned rows.
                    raise RuntimeError(f"non-finite activation for prompt {rid} at position {len(tokens[o.seq_id])}")
                for layer, j in zip(args.position_layers, position_index, strict=True):
                    per_pos[layer][rid, t] = act[j].numpy()
                if t == 0:
                    pos0[o.seq_id] = act
                sums[o.seq_id] = sums.get(o.seq_id, 0) + act
                tokens[o.seq_id].append(o.token_id)
                if not o.finished:
                    continue
                toks = tokens.pop(o.seq_id)
                pooled[rid, 0] = pos0.pop(o.seq_id).numpy()
                pooled[rid, 1] = (sums.pop(o.seq_id) / len(toks)).numpy()
                rec = {
                    **seq_to_record.pop(o.seq_id),
                    "output_token_ids": toks,
                    "response": tokenizer.decode(toks, skip_special_tokens=True),
                    "finish_reason": o.finish_reason,
                }
                out.write(json.dumps(rec) + "\n")
                out.flush()
                finished += 1
                if finished % 100 == 0:
                    rate = finished / (time.perf_counter() - t0)
                    print(f"{finished} done, {rate:.2f} prompts/s, ~{len(seq_to_record) / rate / 60:.0f} min left")
    for arr in [*per_pos.values(), pooled]:
        arr.flush()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
