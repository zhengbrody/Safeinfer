"""Turns next-token logits into tokens: greedy, temperature, top-k, top-p.

Filtering is batched over rows with per-row settings; only the final draw
loops over rows, because each request draws from its own RNG (so a seeded
request samples the same tokens however the scheduler batches it).
"""

from __future__ import annotations

import torch

from safeinfer.sampling_params import SamplingParams


def filtered_probs(
    logits: torch.Tensor,  # [N, V] fp32
    temperature: torch.Tensor,  # [N], > 0
    top_k: torch.Tensor,  # [N], 0 disables
    top_p: torch.Tensor,  # [N], in (0, 1]
) -> torch.Tensor:
    """Sampling distribution after temperature, then top-k, then top-p (HF's order)."""
    vocab = logits.shape[-1]
    sorted_logits, order = (logits / temperature[:, None]).sort(dim=-1, descending=True)
    ranks = torch.arange(vocab, device=logits.device)

    k = torch.where(top_k > 0, top_k, vocab)
    sorted_logits = sorted_logits.masked_fill(ranks >= k[:, None], float("-inf"))

    # Keep the smallest prefix whose mass reaches top_p: drop a token once the
    # tokens ranked above it already hold top_p. The top token is always kept.
    probs = sorted_logits.softmax(dim=-1)
    mass_above = probs.cumsum(dim=-1) - probs
    sorted_logits = sorted_logits.masked_fill(mass_above >= top_p[:, None], float("-inf"))

    filtered = torch.full_like(logits, float("-inf")).scatter(-1, order, sorted_logits)
    return filtered.softmax(dim=-1)


def sample(
    logits: torch.Tensor,  # [N, V] fp32
    params: list[SamplingParams],
    generators: list[torch.Generator | None],
) -> list[int]:
    """One token per row. A None generator draws from torch's global RNG."""
    tokens = logits.argmax(dim=-1)
    rows = [i for i, p in enumerate(params) if not p.greedy]
    if rows:
        device = logits.device
        probs = filtered_probs(
            logits[rows],
            torch.tensor([params[i].temperature for i in rows], device=device),
            torch.tensor([params[i].top_k for i in rows], device=device),
            torch.tensor([params[i].top_p for i in rows], device=device),
        )
        for j, i in enumerate(rows):
            tokens[i] = torch.multinomial(probs[j], 1, generator=generators[i])[0]
    return tokens.tolist()
