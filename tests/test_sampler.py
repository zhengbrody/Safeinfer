from __future__ import annotations

import pytest
import torch
from transformers.generation.logits_process import (
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from safeinfer.engine.sampler import filtered_probs, sample
from safeinfer.sampling_params import SamplingParams


@pytest.mark.parametrize(
    ("temperature", "top_k", "top_p"), [(1.0, 0, 1.0), (0.7, 5, 1.0), (1.3, 0, 0.8), (0.9, 20, 0.6)]
)
def test_filtered_probs_match_hf_warpers(temperature, top_k, top_p):
    logits = torch.randn(4, 64, generator=torch.Generator().manual_seed(0))
    warpers = LogitsProcessorList([TemperatureLogitsWarper(temperature)])
    if top_k:
        warpers.append(TopKLogitsWarper(top_k))
    if top_p < 1:
        warpers.append(TopPLogitsWarper(top_p))
    ref = warpers(torch.zeros(4, 1, dtype=torch.long), logits.clone()).softmax(-1)

    got = filtered_probs(logits, torch.full((4,), temperature), torch.full((4,), top_k), torch.full((4,), top_p))
    torch.testing.assert_close(got, ref)


def test_top_p_boundary_keeps_smallest_set_reaching_p():
    # Exact probabilities 0.5 / 0.25 / 0.25: the top token alone already reaches top_p=0.5.
    logits = torch.tensor([[0.5, 0.25, 0.25]]).log()
    got = filtered_probs(logits, torch.ones(1), torch.zeros(1, dtype=torch.long), torch.tensor([0.5]))
    torch.testing.assert_close(got, torch.tensor([[1.0, 0.0, 0.0]]))
    ref = TopPLogitsWarper(0.5)(torch.zeros(1, 1, dtype=torch.long), logits.clone()).softmax(-1)
    torch.testing.assert_close(got, ref)


def test_filtered_probs_settings_are_per_row():
    logits = torch.randn(2, 32, generator=torch.Generator().manual_seed(1))
    got = filtered_probs(logits, torch.tensor([1.0, 1.0]), torch.tensor([3, 0]), torch.tensor([1.0, 1.0]))
    assert (got[0] > 0).sum() == 3
    assert (got[1] > 0).sum() == 32


def test_greedy_rows_take_argmax_and_ignore_generators():
    logits = torch.randn(3, 50, generator=torch.Generator().manual_seed(2))
    params = [SamplingParams(), SamplingParams(temperature=1.0, top_k=1), SamplingParams()]
    assert sample(logits, params, [None, None, None]) == logits.argmax(-1).tolist()


def test_seeded_rows_are_reproducible_and_independent_of_neighbours():
    logits = torch.randn(3, 50, generator=torch.Generator().manual_seed(3))
    p = SamplingParams(temperature=1.0)

    def draw(rows, seeds):
        gens = [torch.Generator().manual_seed(s) for s in seeds]
        return [sample(logits[rows], [p] * len(rows), gens) for _ in range(5)]

    together = draw([0, 1, 2], [7, 8, 9])
    alone = draw([1], [8])
    assert [t[1] for t in together] == [a[0] for a in alone]
    assert draw([0, 1, 2], [7, 8, 9]) == together
    # A different seed gives a different sequence of draws (over 5 draws from 50 tokens).
    assert [t[1] for t in draw([0, 1, 2], [7, 99, 9])] != [t[1] for t in together]
