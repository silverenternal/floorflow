"""Candidate ranking metrics for constraint-aware gating."""

from __future__ import annotations

import numpy as np


def ndcg_at_k(
    scores: np.ndarray,
    relevance: np.ndarray,
    *,
    k: int = 5,
) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    relevance = np.asarray(relevance, dtype=np.float64)
    if scores.shape != relevance.shape:
        raise ValueError("scores and relevance must match")
    order = np.argsort(-scores, kind="mergesort")[: int(k)]
    gains = relevance[order]
    discounts = 1.0 / np.log2(np.arange(2, len(order) + 2))
    dcg = float((gains * discounts).sum())
    ideal = np.sort(relevance)[::-1][: int(k)]
    ideal_dcg = float(
        (ideal / np.log2(np.arange(2, len(ideal) + 2))).sum()
    )
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


__all__ = ["ndcg_at_k"]
