"""Distributional metrics for conditional floorplan samples."""

from __future__ import annotations

import numpy as np


def pairwise_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.linalg.norm(left[:, None] - right[None], axis=-1)


def mean_pairwise_distance(values: np.ndarray) -> float:
    """Return mean distance over unique pairs, excluding self-distance."""

    if len(values) < 2:
        return 0.0
    distances = pairwise_distances(values, values)
    return float(distances[np.triu_indices(len(values), k=1)].mean())


def mmd_rbf(left: np.ndarray, right: np.ndarray) -> float:
    joined = np.concatenate([left, right], axis=0)
    pairwise = pairwise_distances(joined, joined)
    positive = pairwise[pairwise > 0]
    bandwidth = float(np.median(positive)) if positive.size else 1.0
    bandwidth = max(bandwidth, 1.0e-6)

    def kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        distance = pairwise_distances(a, b)
        return np.exp(-(distance**2) / (2.0 * bandwidth**2))

    return float(
        kernel(left, left).mean()
        + kernel(right, right).mean()
        - 2.0 * kernel(left, right).mean()
    )


def coverage_density(
    generated: np.ndarray,
    target: np.ndarray,
) -> dict[str, float]:
    distance = pairwise_distances(target, generated)
    target_pairwise = pairwise_distances(target, target)
    positive = target_pairwise[target_pairwise > 0]
    radius = float(np.median(positive)) if positive.size else 1.0
    return {
        "coverage": float((distance.min(axis=1) <= radius).mean()),
        "density": float((distance <= radius).sum(axis=1).mean()),
    }


def bootstrap_mean(
    values: np.ndarray,
    *,
    seed: int,
    samples: int = 1000,
) -> dict[str, float]:
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
    }


__all__ = [
    "bootstrap_mean",
    "coverage_density",
    "mean_pairwise_distance",
    "mmd_rbf",
    "pairwise_distances",
]
