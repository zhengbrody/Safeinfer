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
├── attention/
│   ├── backend.py        AttentionMetadata + AttentionBackend protocol
│   └── dense.py          cache-free causal attention (correctness reference)
├── model/
│   ├── llama.py          hand-written Llama/Qwen2 forward; HF-identical param names
│   └── loader.py         safetensors -> model (meta-device init, tied embeddings)
└── bench/
    ├── workload.py       Request, Poisson arrivals, ShareGPT length sampling
    ├── metrics.py        RequestTiming -> throughput, TTFT/TPOT/E2E p50/p99
    ├── hf_baseline.py    HF generate with static batching, real-time replay
    └── run_bench.py      CLI: one backend + one workload -> result JSON
```

### Key design decisions

- **Flattened token batch.** `model.forward(input_ids[T], positions[T], backend, meta)`
  takes all sequences concatenated; `AttentionMetadata` says where each one
  is. Mixing prefill and decode sequences in one step (continuous batching,
  chunked prefill) therefore needs no model changes; only backends and
  metadata grow.
- **Attention is a pluggable backend.** The model owns projections and RoPE;
  the backend owns KV storage and the attention op. Week 2 adds paged
  backends alongside `DenseAttention`.
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
ruff check . && ruff format --check .
python -m safeinfer.bench.run_bench --backend hf --model <hf-id-or-dir> \
    [--dataset sharegpt.json] --num-requests 200 --rate 4 --out results/hf_r4.json
```

## Roadmap / status

- [x] **Week 1: model, baseline, measurement**
  - [x] Llama/Qwen2 forward (RMSNorm, RoPE incl. llama3 scaling, GQA, SwiGLU, qkv bias)
  - [x] safetensors loader; logits, hidden states, and 20-step greedy decode match HF
  - [x] workload generator, metrics, HF static-batching baseline, bench CLI, CI
  - [ ] run the baseline on a GPU with the target model (needs GPU access)
- [ ] **Week 2: paged KV cache.** `memory/block_manager.py` (+ hypothesis property
      tests), `memory/kv_cache.py`, `attention/torch_paged.py`; greedy matches HF
- [ ] **Week 3: continuous batching.** `scheduler/` (iteration-level, recompute
      preemption, token budget), `engine/engine.py`, `engine/sampler.py`; rate sweep
- [ ] **Week 4: safety probes.** Label = did the generation comply (judge-labelled),
      not prompt topic; separate input vs output probes; OOD split by attack family;
      over-refusal check; streaming-moderation baseline
- [ ] **Week 5: speed + write-up.** CUDA graphs for decode first, then a Triton
      paged-decode kernel; profiler-backed gap analysis vs vLLM

## Changelog

- 2026-10-20: Project scaffold. Week-1 model, loader, parity tests, and bench tooling done.
