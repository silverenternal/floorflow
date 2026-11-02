"""Reusable diagnostics for target-free rho candidate ranking."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def candidate_rank_metrics(
    predictions: list[np.ndarray],
    labels: list[np.ndarray],
) -> dict[str, float | int]:
    """Return pairwise, top-1 and per-case rank agreement."""

    if len(predictions) != len(labels) or not predictions:
        raise ValueError("predictions and labels must be non-empty and aligned")
    pairs = correct = top1 = 0
    correlations = []
    for predicted, actual in zip(predictions, labels, strict=True):
        if predicted.shape != actual.shape or predicted.ndim != 1:
            raise ValueError("candidate predictions and labels must be one-dimensional")
        top1 += int(np.argmin(predicted) == np.argmin(actual))
        if np.std(predicted) > 0 and np.std(actual) > 0:
            correlations.append(float(spearmanr(predicted, actual).statistic))
        for left in range(len(actual)):
            for right in range(left + 1, len(actual)):
                pairs += 1
                correct += int(
                    (predicted[left] < predicted[right])
                    == (actual[left] < actual[right])
                )
    return {
        "cases": len(labels),
        "pair_accuracy": correct / max(pairs, 1),
        "top1_accuracy": top1 / len(labels),
        "mean_case_spearman": float(np.mean(correlations)) if correlations else 0.0,
    }


def case_flags(
    candidate: dict[str, float],
    reference: dict[str, float],
) -> tuple[bool, bool, bool]:
    """Return joint, strict structural and constraint-safe comparisons."""

    joint = (
        candidate["legalized_hard_cost"] <= reference["legalized_hard_cost"]
        and candidate["legalized_v_rel"] <= reference["legalized_v_rel"]
    )
    strict = (
        joint
        and candidate["legalized_grouping"] <= reference["legalized_grouping"]
        and candidate["legalized_boundary"] <= reference["legalized_boundary"]
        and candidate["legalized_p2b_hpwl_gap"]
        <= reference["legalized_p2b_hpwl_gap"]
    )
    safe = (
        strict
        and candidate["legalized_area_gap"] <= reference["legalized_area_gap"]
        and candidate["legalized_b2b_hpwl_gap"]
        <= reference["legalized_b2b_hpwl_gap"]
    )
    return joint, strict, safe


__all__ = ["candidate_rank_metrics", "case_flags"]
