from __future__ import annotations

import pytest

from floorflow.evaluation.paired_bootstrap import (
    clustered_paired_delta,
    paired_delta,
    paired_effect_passes,
)


def test_paired_delta_reports_improvements() -> None:
    current = [{"x": 2.0}, {"x": 1.0}, {"x": 3.0}]
    baseline = [{"x": 1.0}, {"x": 2.0}, {"x": 1.0}]
    result = paired_delta(current, baseline, "x", iterations=100)
    assert result["cases"] == 3
    assert result["mean_delta"] == pytest.approx(2.0 / 3.0)
    assert result["improved"] == 2
    assert result["worsened"] == 1
    assert result["rank_biserial_difference"] > 0.0
    assert result["cliffs_delta_difference"] == pytest.approx(1.0 / 3.0)
    assert result["wilcoxon_p_greater"] < 0.5
    assert result["effect_size_method"] == "paired_difference_vs_zero"


def test_paired_delta_rejects_unpaired_rows() -> None:
    with pytest.raises(ValueError, match="equal lengths"):
        paired_delta([{"x": 1.0}], [], "x")


def test_paired_delta_respects_lower_is_better_metrics() -> None:
    result = paired_delta(
        [{"cost": 1.0}, {"cost": 3.0}],
        [{"cost": 2.0}, {"cost": 2.0}],
        "cost",
        iterations=20,
        lower_is_better=True,
    )

    assert result["improved"] == 1
    assert result["worsened"] == 1


def test_paired_delta_aligns_unique_case_indices() -> None:
    current = [{"index": 1, "cost": 3.0}, {"index": 0, "cost": 1.0}]
    baseline = [{"index": 0, "cost": 2.0}, {"index": 1, "cost": 4.0}]

    result = paired_delta(
        current,
        baseline,
        "cost",
        iterations=20,
        lower_is_better=True,
    )

    assert result["mean_delta"] == pytest.approx(-1.0)
    assert result["improved"] == 2


def test_paired_delta_rejects_mismatched_case_indices() -> None:
    with pytest.raises(ValueError, match="mismatched case indices"):
        paired_delta(
            [{"index": 0, "cost": 1.0}],
            [{"index": 1, "cost": 1.0}],
            "cost",
            lower_is_better=True,
        )


def test_clustered_paired_delta_collapses_repeated_cases() -> None:
    baseline = [
        {"index": 0, "x": 10.0},
        {"index": 0, "x": 12.0},
        {"index": 1, "x": 20.0},
        {"index": 1, "x": 22.0},
    ]
    current = [
        {"index": 0, "x": 8.0},
        {"index": 0, "x": 10.0},
        {"index": 1, "x": 21.0},
        {"index": 1, "x": 23.0},
    ]

    result = clustered_paired_delta(
        current,
        baseline,
        "x",
        iterations=100,
        seed=1,
        lower_is_better=True,
    )

    assert result["cases"] == 2
    assert result["rows"] == 4
    assert result["mean_delta"] == pytest.approx(-0.5)
    assert result["improved"] == 1
    assert result["worsened"] == 1
    higher = clustered_paired_delta(
        current,
        baseline,
        "x",
        iterations=100,
        seed=1,
        lower_is_better=False,
    )
    assert higher["improved"] == 1
    assert higher["worsened"] == 1
    positive_current = [{"index": 0, "x": 2.0}, {"index": 1, "x": 4.0}]
    positive_baseline = [{"index": 0, "x": 1.0}, {"index": 1, "x": 2.0}]
    positive = clustered_paired_delta(
        positive_current,
        positive_baseline,
        "x",
        iterations=100,
        seed=1,
        lower_is_better=False,
    )
    assert positive["improved"] == 2
    assert positive["worsened"] == 0


def test_clustered_paired_delta_rejects_mismatched_clusters() -> None:
    with pytest.raises(ValueError, match="mismatched"):
        clustered_paired_delta(
            [{"index": 1, "x": 1.0}],
            [{"index": 2, "x": 1.0}],
            "x",
        )


def test_paired_effect_gate_handles_ceiling_and_direction() -> None:
    assert paired_effect_passes(
        {"ci_low": 0.1, "ci_high": 0.2, "baseline_mean": 0.5}
    )
    assert paired_effect_passes(
        {"ci_low": 0.0, "ci_high": 0.0, "baseline_mean": 1.0},
        ceiling=1.0,
    )
    assert not paired_effect_passes(
        {"ci_low": 0.0, "ci_high": 0.0, "baseline_mean": 0.9},
        ceiling=1.0,
    )
    assert paired_effect_passes(
        {"ci_low": -0.2, "ci_high": -0.1, "baseline_mean": 2.0},
        lower_is_better=True,
    )
    assert not paired_effect_passes(
        {"ci_low": -0.2, "ci_high": 0.1, "baseline_mean": 2.0},
        lower_is_better=True,
    )
