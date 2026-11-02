"""Reusable paired bootstrap statistics for case-level evaluation rows."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.stats import rankdata, wilcoxon


def _effect_sizes(differences: np.ndarray) -> dict[str, float]:
    nonzero = differences[differences != 0.0]
    positive = int((differences > 0.0).sum())
    negative = int((differences < 0.0).sum())
    if nonzero.size == 0:
        return {
            "rank_biserial_difference": 0.0,
            "cliffs_delta_difference": 0.0,
            "wilcoxon_p_greater": 1.0,
            "wilcoxon_p_two_sided": 1.0,
        }
    ranks = rankdata(np.abs(nonzero), method="average")
    positive_rank = float(ranks[nonzero > 0.0].sum())
    negative_rank = float(ranks[nonzero < 0.0].sum())
    denominator = positive_rank + negative_rank
    rank_biserial = (
        (positive_rank - negative_rank) / denominator
        if denominator > 0.0
        else 0.0
    )
    return {
        "rank_biserial_difference": float(rank_biserial),
        "cliffs_delta_difference": float(
            (positive - negative) / differences.size
        ),
        "wilcoxon_p_greater": float(
            wilcoxon(differences, alternative="greater").pvalue
        ),
        "wilcoxon_p_two_sided": float(
            wilcoxon(differences, alternative="two-sided").pvalue
        ),
    }


def paired_delta(
    current: Sequence[dict[str, Any]],
    baseline: Sequence[dict[str, Any]],
    key: str,
    iterations: int = 1000,
    seed: int = 17,
    *,
    lower_is_better: bool = False,
) -> dict[str, float | int | str]:
    if len(current) != len(baseline):
        raise ValueError("paired rows must have equal lengths")
    if not current:
        raise ValueError("paired rows must be non-empty")
    if all("index" in row for row in current) and all(
        "index" in row for row in baseline
    ):
        current_indices = [int(row["index"]) for row in current]
        baseline_indices = [int(row["index"]) for row in baseline]
        if len(set(current_indices)) == len(current_indices):
            if set(current_indices) != set(baseline_indices):
                raise ValueError("paired rows have mismatched case indices")
            baseline_by_index = {
                int(row["index"]): row for row in baseline
            }
            current = sorted(current, key=lambda row: int(row["index"]))
            baseline = [
                baseline_by_index[int(row["index"])] for row in current
            ]
        elif current_indices != baseline_indices:
            raise ValueError(
                "paired rows with repeated indices must have identical order"
            )
    differences = np.asarray(
        [
            float(cur[key]) - float(base[key])
            for cur, base in zip(current, baseline, strict=True)
        ],
        dtype=np.float64,
    )
    baseline_values = np.asarray(
        [float(base[key]) for base in baseline],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    sampled = differences[
        rng.integers(0, len(differences), (max(1, iterations), len(differences)))
    ]
    means = sampled.mean(axis=1)
    improved = differences < 0.0 if lower_is_better else differences > 0.0
    worsened = differences > 0.0 if lower_is_better else differences < 0.0
    return {
        "cases": len(differences),
        "mean_delta": float(differences.mean()),
        "baseline_mean": float(baseline_values.mean()),
        "relative_delta": (
            float(differences.mean() / abs(baseline_values.mean()))
            if baseline_values.mean() != 0.0
            else None
        ),
        "median_delta": float(np.median(differences)),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "ci_method": "paired_percentile_bootstrap",
        "ci_iterations": int(max(1, iterations)),
        "ci_seed": int(seed),
        "improved": int(improved.sum()),
        "worsened": int(worsened.sum()),
        "unchanged": int((differences == 0).sum()),
        **_effect_sizes(differences),
        "effect_size_method": "paired_difference_vs_zero",
    }


def clustered_paired_delta(
    current: Sequence[dict[str, Any]],
    baseline: Sequence[dict[str, Any]],
    key: str,
    cluster_key: str = "index",
    iterations: int = 1000,
    seed: int = 17,
    lower_is_better: bool = True,
) -> dict[str, float | int | str]:
    """Bootstrap paired effects after collapsing repeated rows by cluster."""

    if len(current) != len(baseline):
        raise ValueError("paired rows must have equal lengths")
    if not current:
        raise ValueError("paired rows must be non-empty")
    differences: dict[Any, list[float]] = {}
    baseline_values: dict[Any, list[float]] = {}
    for cur, base in zip(current, baseline, strict=True):
        if cur[cluster_key] != base[cluster_key]:
            raise ValueError("paired rows have mismatched cluster identifiers")
        cluster = cur[cluster_key]
        differences.setdefault(cluster, []).append(float(cur[key]) - float(base[key]))
        baseline_values.setdefault(cluster, []).append(float(base[key]))
    clusters = sorted(differences, key=str)
    cluster_delta = np.asarray(
        [np.mean(differences[cluster]) for cluster in clusters],
        dtype=np.float64,
    )
    cluster_baseline = np.asarray(
        [np.mean(baseline_values[cluster]) for cluster in clusters],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    sampled = cluster_delta[
        rng.integers(
            0,
            len(cluster_delta),
            (max(1, iterations), len(cluster_delta)),
        )
    ]
    means = sampled.mean(axis=1)
    baseline_mean = float(cluster_baseline.mean())
    effect = float(cluster_delta.mean())
    return {
        "cases": int(cluster_delta.size),
        "rows": len(current),
        "mean_delta": effect,
        "baseline_mean": baseline_mean,
        "relative_delta": effect / abs(baseline_mean) if baseline_mean != 0.0 else None,
        "median_delta": float(np.median(cluster_delta)),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "ci_method": "cluster_paired_percentile_bootstrap",
        "cluster_key": cluster_key,
        "ci_iterations": int(max(1, iterations)),
        "ci_seed": int(seed),
        "improved": int(
            (
                (cluster_delta < 0.0)
                if lower_is_better
                else (cluster_delta > 0.0)
            ).sum()
        ),
        "worsened": int(
            (
                (cluster_delta > 0.0)
                if lower_is_better
                else (cluster_delta < 0.0)
            ).sum()
        ),
        "unchanged": int((cluster_delta == 0.0).sum()),
        "lower_is_better": bool(lower_is_better),
        **_effect_sizes(cluster_delta),
        "effect_size_method": "clustered_paired_difference_vs_zero",
    }


def paired_effect_passes(
    result: dict[str, Any],
    *,
    lower_is_better: bool = False,
    ceiling: float | None = None,
    tolerance: float = 1.0e-12,
) -> bool:
    """Apply a strict paired gate while allowing non-regression at a ceiling."""

    ci_low = float(result["ci_low"])
    ci_high = float(result["ci_high"])
    if lower_is_better:
        return ci_high < -float(tolerance)
    if ci_low > float(tolerance):
        return True
    return (
        ceiling is not None
        and float(result["baseline_mean"]) >= float(ceiling) - float(tolerance)
        and ci_low >= -float(tolerance)
    )


__all__ = [
    "clustered_paired_delta",
    "paired_delta",
    "paired_effect_passes",
]
