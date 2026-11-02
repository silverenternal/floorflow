"""Effect-size helpers for paired and unpaired metric comparisons."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.stats import rankdata


def rank_biserial_correlation(differences: Sequence[float]) -> float:
    """Return the signed-rank paired effect size for non-zero differences."""

    values = np.asarray(differences, dtype=np.float64)
    values = values[values != 0.0]
    if values.size == 0:
        return 0.0
    ranks = rankdata(np.abs(values), method="average")
    positive = float(ranks[values > 0].sum())
    negative = float(ranks[values < 0].sum())
    denominator = positive + negative
    return (positive - negative) / denominator if denominator else 0.0


def cliffs_delta(left: Sequence[float], right: Sequence[float]) -> float:
    """Return Cliff's delta: P(left > right) - P(left < right)."""

    left_values = np.asarray(left, dtype=np.float64)
    right_values = np.asarray(right, dtype=np.float64)
    if left_values.size == 0 or right_values.size == 0:
        return 0.0
    differences = (
        np.sign(left_values[:, None] - right_values[None, :]).astype(np.float64)
    )
    return float(differences.mean())


__all__ = ["cliffs_delta", "rank_biserial_correlation"]
