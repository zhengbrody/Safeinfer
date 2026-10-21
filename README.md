# safeinfer

An LLM inference engine written from scratch in PyTorch. It has a paged KV
cache and continuous batching, and runs linear safety probes on activations
from the same forward pass, so risky generations can be flagged or stopped
mid-decode at almost no extra cost.

**Status:** week 4 of 5. The hand-written Llama/Qwen2 forward, paged KV
cache, and continuous-batching scheduler (chunked prefill, recompute
preemption) reproduce HuggingFace token for token (real Llama-3.2-1B
included). On 1000 ShareGPT conversations, paging cuts KV waste from 80.6%
to 1.8% and fits 4.4x more concurrent requests in the same memory. A
probe on the prompt's activations spots harmful requests at least as well as
Llama Guard 3 1B (AUROC 0.992 vs 0.967) while flagging 0.8% of XSTest's safe
prompts (Guard: 21% at its default threshold), for one dot product per
request. See
[CLAUDE.md](CLAUDE.md) for architecture, results, and roadmap.

```bash
pip install -e ".[dev]"
pytest -q
python -m safeinfer.bench.run_bench --backend engine --model meta-llama/Llama-3.2-1B-Instruct \
    --num-requests 200 --rate 4 --out results/engine_r4.json
```
