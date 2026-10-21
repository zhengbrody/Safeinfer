# safeinfer — engineering notes

LLM inference engine built from scratch: paged KV cache + continuous batching,
with linear safety probes that read activations from the same forward pass
(no extra forward). The project plan lives in the Claude Doc "Safety-Monitored
LLM Inference Engine — 项目框架"; this file records how the code is built and
what is done.

## Principles

- **Correct first, then fast.** Every engine path is checked against HF
  (`tests/test_llama_parity.py`). A new attention backend must match
  `DenseAttention` before it is benchmarked.
- **No stub files.** A module is created in the week it gets implemented; the
  roadmap below is the plan, not empty files.
- **transformers is never an engine dependency.** It is only a test oracle and
  the benchmark baseline (`[bench]`/`[dev]` extras).
- Every benchmark result JSON records args, library versions, and GPU.

## Architecture

```
safeinfer/
├── config.py             ModelConfig parsed from raw config.json (llama, qwen2)
├── sampling_params.py    SamplingParams: max_tokens, temperature/top-k/top-p, seed, stop ids
├── attention/
│   ├── backend.py        AttentionMetadata (query/context lens, slots, block tables) + protocol
│   ├── dense.py          cache-free causal attention (correctness reference)
│   └── torch_paged.py    paged KV: write slots, gather blocks, SDPA with offset causal mask
├── memory/
│   ├── block_manager.py  free list + per-sequence block tables; grow/free; invariants
│   └── kv_cache.py       physical K/V blocks [layers, blocks, block_size, kv_heads, dim]; sizing
├── model/
│   ├── llama.py          hand-written Llama/Qwen2 forward; HF-identical param names
│   └── loader.py         safetensors -> model (meta-device init, tied embeddings)
├── scheduler/
│   ├── sequence.py       Sequence state: tokens, num_computed_tokens, status, per-request RNG
│   └── scheduler.py      iteration-level FCFS scheduler: token budget, chunked prefill, recompute preemption
├── engine/
│   ├── model_runner.py   SeqStep batch -> flattened forward on the paged cache -> logits
│   ├── static_batch.py   static batching on the paged engine (experiment group "engine A")
│   ├── sampler.py        batched temperature/top-k/top-p filtering, per-request RNG draws
│   └── engine.py         LLMEngine: schedule -> run -> sample -> update, one step at a time
└── bench/
    ├── workload.py       Request, Poisson arrivals, ShareGPT length sampling
    ├── metrics.py        RequestTiming -> throughput, TTFT/TPOT/E2E p50/p99
    ├── static_replay.py  real-time static-batching server loop shared by all static backends
    ├── hf_baseline.py    HF generate as a static_replay backend
    ├── engine_static.py  our engine as a static_replay backend; KV-capacity admission
    ├── continuous_replay.py real-time loop for the continuous engine (requests join every step)
    ├── kv_utilization.py contiguous vs paged KV waste/concurrency on real lengths (no GPU)
    └── run_bench.py      CLI: one backend + one workload -> result JSON
```

### Key design decisions

- **Flattened token batch.** `model.forward(input_ids[T], positions[T], backend, meta)`
  takes all sequences concatenated; `AttentionMetadata` says where each one
  is. Mixing prefill and decode sequences in one step (continuous batching,
  chunked prefill) therefore needs no model changes; only backends and
  metadata grow.
- **Attention is a pluggable backend.** The model owns projections and RoPE;
  the backend owns KV storage and the attention op.
- **One metadata shape for every step type.** A sequence's new tokens are the
  last `query_len` of its `context_len` tokens: prefill, decode, and chunked
  prefill are the same code path. SDPA's `is_causal` aligns top-left, so the
  paged backend builds an offset causal mask explicitly.
- **BlockManager is bookkeeping only** (no tensors): `grow(seq, n)` returns the
  new tokens' slots (`block * block_size + offset`), `free(seq)` returns
  blocks. `ModelRunner.step` checks capacity for the whole batch first, so a
  step either fully happens or changes nothing.
- **Static admission reserves worst case** (prompt + full output) because
  static mode cannot preempt; continuous batching admits optimistically.
- **Scheduler policy (vLLM-like FCFS).** Every step: running sequences first in
  admission order (decode = 1 token; mid chunked prefill = rest of prompt, capped
  by the token budget); if the pool cannot hold one, evict the *newest* runner;
  only if nothing was evicted, admit waiting requests in order. Admission is
  optimistic (blocks for this step's tokens only); preemption absorbs the misses.
- **Preemption is recompute, not swap.** A victim frees all blocks, keeps its
  generated tokens, and goes to the *front* of the queue; it later re-prefills
  prompt + outputs. Config checks (pool holds one max_model_len sequence; without
  chunking, budget >= max_model_len) guarantee the queue head always fits an
  empty engine, so the scheduler cannot livelock. Week 4 note: a recompute
  re-runs the probe over old tokens, so any k-consecutive-flag counter must be
  keyed to output positions, not to forward passes.
- **Scheduler books, runner allocates.** The scheduler only counts blocks
  (`blocks_needed`) and frees on preempt/finish; `ModelRunner.step` grows. One
  owner per mutation keeps the hypothesis invariants simple.
- **Per-request RNG.** A seeded request draws from its own `torch.Generator`, so
  its samples are identical alone, batched, or after preemption (tested).
  Filtering is batched; only the final multinomial draw loops over rows.
- **Stop tokens are kept in the output**; `ignore_eos` (benchmarks) disables
  model EOS but not explicit `stop_token_ids`.
- **Probe activations come from the same forward.** `capture_layer=L` returns the
  residual stream after layer L (= HF `hidden_states[L+1]`).
- **Logits only where needed.** `forward` returns hidden states;
  `compute_logits` runs on just the rows being sampled (the last token of
  each sequence), never the whole prefill.
- **Separate q/k/v and gate/up projections**, mirroring HF so checkpoints load
  as-is. Fusing them is a later optimization (it must stay parity-tested).
- **HF baseline is credited with streaming timings** (per-step timestamps via
  a StoppingCriteria) and only requested tokens count toward throughput,
  so the baseline is not a strawman.

### Numerics

RMSNorm and RoPE run in fp32 and cast back, matching HF op order. The parity
tests run in fp32 on CPU with every parameter randomized (HF zero-inits
biases, which would hide loading bugs). In bf16, batching changes reduction
order, so expect logit drift instead of token-identical output on long
sequences; compare logits with a tolerance there.

## Commands

```bash
pip install -e ".[dev]"
pytest -q                     # CPU, ~5s
SAFEINFER_TEST_MODEL=meta-llama/Llama-3.2-1B-Instruct pytest tests/test_real_checkpoint.py  # ~10 GB RAM
ruff check . && ruff format --check .
python -m safeinfer.bench.run_bench --backend {hf,engine-static,engine} --model <hf-id-or-dir> \
    [--dataset data/sharegpt.json] --num-requests 200 --rate 4 --out results/<name>.json
    # engine: --max-batch-size = max running seqs; --max-num-batched-tokens; --enable-chunked-prefill
# GPU rate sweep (same workload, three backends):
for r in 1 2 4 8 16 inf; do for b in hf engine-static engine; do
  python -m safeinfer.bench.run_bench --backend $b --model meta-llama/Llama-3.2-1B-Instruct \
    --dataset data/sharegpt.json --num-requests 500 --rate $r --max-batch-size 64 --out results/gpu_${b}_r$r.json
done; done
python -m safeinfer.bench.kv_utilization --dataset data/sharegpt.json \
    --model meta-llama/Llama-3.2-1B-Instruct --kv-budget-gib 16
```

`data/` (ShareGPT, 650 MB) and `results/` are gitignored. ShareGPT:
https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered
(`ShareGPT_V3_unfiltered_cleaned_split.json`, saved as `data/sharegpt.json`).

## Results so far

KV utilization, 1000 ShareGPT requests (final length mean 456 / p50 374),
Llama-3.2-1B (32 KiB/token bf16), 16 GiB KV budget, max_model_len 2048:

| scheme | lifetime utilization | waste | max concurrent (memory bound) |
|---|---|---|---|
| contiguous, reserve 2048 | 19.4% | 80.6% | 256 |
| paged, block 8 | 99.1% | 0.9% | 1,142 |
| paged, block 16 | 98.2% | 1.8% | 1,132 |
| paged, block 32 | 96.3% | 3.7% | 1,112 |

Concurrency here is a memory bound; compute saturates well before ~1,100.

CPU smoke run (Mac, fp32, Llama-3.2-1B, 16 ShareGPT requests <= 384 tokens, all
at t=0, batch 8). Directional only; real numbers need a GPU:

| backend | output tok/s | p50 TTFT | p50 E2E |
|---|---|---|---|
| hf (static) | 29.7 | 24.2 s | 45.7 s |
| engine-static | 37.5 | 16.2 s | 31.8 s |
| engine (continuous) | 48.5 | 1.6 s | 24.8 s |

## Roadmap / status

- [x] **Week 1: model, baseline, measurement**
  - [x] Llama/Qwen2 forward (RMSNorm, RoPE incl. llama3 scaling, GQA, SwiGLU, qkv bias)
  - [x] safetensors loader; logits, hidden states, and 20-step greedy decode match HF
  - [x] real Llama-3.2-1B-Instruct (fp32, CPU): logits bit-identical to HF (max diff 0), 24-step greedy identical
  - [x] workload generator, metrics, HF static-batching baseline, bench CLI, CI
  - [ ] run the baseline on a GPU with the target model (needs GPU access)
- [x] **Week 2: paged KV cache**
  - [x] BlockManager + stateful hypothesis tests (no leaks, no shared slots, exact counts)
  - [x] KVCache, TorchPagedAttention, ModelRunner, static-batch generator
  - [x] batched greedy == HF (tiny models + real Llama-3.2-1B); chunked prefill and
        mixed prefill/decode steps == dense; mutation-checked
  - [x] `engine-static` bench backend with KV-capacity admission
  - [x] KV utilization analysis on ShareGPT (table above)
  - [ ] GPU run: HF vs engine-static throughput + max batch at equal memory
- [x] **Week 3: continuous batching**
  - [x] scheduler: iteration-level FCFS, token budget, chunked prefill, recompute preemption
  - [x] sampler (temperature/top-k/top-p == HF warpers, incl. exact top-p boundary), per-request seeds
  - [x] LLMEngine; greedy == HF with staggered arrivals + forced preemption, chunked and not
        (tiny models and real Llama-3.2-1B incl. EOS stop); seeded sampling batch-invariant
  - [x] hypothesis workloads: budget/max_num_seqs respected, no leaks, KV == computed tokens,
        every request finishes (~half the random cases preempt); mutation-checked
  - [x] `engine` bench backend + continuous replay loop
  - [ ] GPU rate sweep: hf vs engine-static vs engine (commands above)
- [ ] **Week 4: safety probes.** Label = did the generation comply (judge-labelled),
      not prompt topic; separate input vs output probes; OOD split by attack family;
      over-refusal check; streaming-moderation baseline
- [ ] **Week 5: speed + write-up.** CUDA graphs for decode first, then a Triton
      paged-decode kernel; profiler-backed gap analysis vs vLLM

## Changelog

- 2026-10-20: Project scaffold. Week-1 model, loader, parity tests, and bench tooling done.
- 2026-10-20: Real-checkpoint parity test (opt-in via SAFEINFER_TEST_MODEL); Llama-3.2-1B bit-identical to HF.
  Ruff pinned with explicit rule set so local and CI lint agree.
- 2026-10-20: Week 2 paged KV cache done (CPU-verified). Static replay loop factored out of
  the HF baseline so all static backends share arrival/timing logic. KV analysis: 80.6% -> 1.8%
  waste, 4.4x concurrency at equal memory.
- 2026-10-20: Week 3 continuous batching done (CPU-verified): scheduler, sampler, LLMEngine,
  `engine` bench backend. run_bench model/KV setup shared by both engine backends. CPU smoke:
  continuous 1.6x HF output throughput, p50 TTFT 24 s -> 1.6 s.
