import numpy as np

from floorflow.evaluation.sample_diversity import (
    bootstrap_mean,
    coverage_density,
    mean_pairwise_distance,
)


def test_coverage_and_bootstrap_are_deterministic() -> None:
    target = np.array([[0.0], [1.0], [2.0]])
    generated = np.array([[0.1], [1.1], [2.1], [1.0]])
    metrics = coverage_density(generated, target)
    assert metrics["coverage"] == 1.0
    assert metrics["density"] > 0
    first = bootstrap_mean(np.array([0.1, 0.2, 0.3]), seed=7)
    second = bootstrap_mean(np.array([0.1, 0.2, 0.3]), seed=7)
    assert first == second
    assert first["ci_low"] <= first["mean"] <= first["ci_high"]


def test_diversity_excludes_self_distances() -> None:
    values = np.array([[0.0], [1.0], [2.0]])
    assert np.isclose(mean_pairwise_distance(values), 4.0 / 3.0)
