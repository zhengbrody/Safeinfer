# safeinfer

An LLM inference engine written from scratch in PyTorch. It has a paged KV
cache and continuous batching, and runs linear safety probes on activations
from the same forward pass, so risky generations can be flagged or stopped
mid-decode at almost no extra cost.

**Status:** week 1 of 5. The hand-written Llama/Qwen2 forward matches
HuggingFace, and the HF static-batching baseline and benchmark harness are
done. See [CLAUDE.md](CLAUDE.md) for architecture and roadmap.

```bash
pip install -e ".[dev]"
pytest -q
python -m safeinfer.bench.run_bench --backend hf --model meta-llama/Llama-3.2-1B-Instruct \
    --num-requests 200 --rate 4 --out results/hf_r4.json
```
