"""Generation-level detection metrics for per-position probe scores (numpy only).

The online rule flags a generation at the first position where `k`
consecutive scores reach the threshold. Equivalently, its k-run score
max_t min(scores[t-k+1 .. t]) reaches the threshold; that one number per
generation is what ROC curves and threshold calibration use, so offline
metrics describe exactly the deployed rule.
"""

from __future__ import annotations

import numpy as np


def run_score(scores: np.ndarray, k: int) -> float:
    """Largest value the k-consecutive rule could be thresholded at and still fire.

    Generations shorter than k can never fire: -inf.
    """
    if len(scores) < k:
        return float("-inf")
    windows = np.lib.stride_tricks.sliding_window_view(scores, k)
    return float(windows.min(axis=1).max())


def first_flag(scores: np.ndarray, k: int, threshold: float) -> int | None:
    """First position at which the k-consecutive rule fires, or None."""
    run = 0
    for t, s in enumerate(scores):
        run = run + 1 if s >= threshold else 0
        if run >= k:
            return t
    return None


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve via the rank-sum statistic; ties count half."""
    scores, labels = np.asarray(scores, dtype=float), np.asarray(labels, dtype=bool)
    n_pos, n_neg = labels.sum(), (~labels).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    sorted_scores = scores[order]
    # Average ranks over ties.
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def threshold_at_fpr(negative_scores: np.ndarray, fpr: float) -> float:
    """Smallest threshold at which at most `fpr` of negatives score >= it."""
    neg = np.sort(np.asarray(negative_scores, dtype=float))[::-1]
    allowed = int(np.floor(fpr * len(neg)))  # negatives allowed at or above the threshold
    if allowed >= len(neg):
        return float(neg[-1])
    # Strictly above the (allowed+1)-th highest negative.
    return float(np.nextafter(neg[allowed], np.inf))


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """95% Wilson score interval for a binomial rate; None when n == 0."""
    if n == 0:
        return None
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (float(center - half), float(center + half))
