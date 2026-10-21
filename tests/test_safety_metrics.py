from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from safeinfer.safety.metrics import auroc, first_flag, run_score, threshold_at_fpr, wilson_interval

scores_st = st.lists(st.floats(0, 1, allow_nan=False), min_size=1, max_size=30).map(np.array)


@given(scores=scores_st, k=st.integers(1, 4), threshold=st.floats(0, 1))
def test_run_score_agrees_with_online_rule(scores, k, threshold):
    # The rule fires at some position exactly when the run score reaches the threshold.
    assert (first_flag(scores, k, threshold) is not None) == (run_score(scores, k) >= threshold)


def test_first_flag_needs_consecutive_hits():
    s = np.array([0.9, 0.1, 0.9, 0.9, 0.2])
    assert first_flag(s, 1, 0.5) == 0
    assert first_flag(s, 2, 0.5) == 3
    assert first_flag(s, 3, 0.5) is None
    assert run_score(s, 2) == 0.9 and run_score(s, 6) == float("-inf")


def test_auroc_hand_computed_with_ties():
    assert auroc(np.array([0.1, 0.4, 0.35, 0.8]), np.array([0, 0, 1, 1])) == pytest.approx(0.75)
    assert auroc(np.array([0.5, 0.5]), np.array([0, 1])) == pytest.approx(0.5)
    assert auroc(np.array([1.0, 2.0]), np.array([1, 0])) == 0.0


@given(neg=st.lists(st.floats(0, 1), min_size=1, max_size=200).map(np.array), fpr=st.floats(0, 1))
def test_threshold_at_fpr_respects_budget(neg, fpr):
    t = threshold_at_fpr(neg, fpr)
    assert (neg >= t).mean() <= fpr
    if fpr < 1:  # and it is the smallest such threshold
        assert (neg >= np.nextafter(t, -np.inf)).mean() > fpr


def test_wilson_interval_known_values():
    lo, hi = wilson_interval(8, 20)  # p = 0.4: standard reference values
    assert lo == pytest.approx(0.2188, abs=1e-3) and hi == pytest.approx(0.6134, abs=1e-3)
    assert wilson_interval(0, 10)[0] == pytest.approx(0.0, abs=1e-12) and wilson_interval(0, 0) is None
