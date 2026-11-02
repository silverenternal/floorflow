from __future__ import annotations

from floorflow.evaluation.statistical_effect_sizes import (
    cliffs_delta,
    rank_biserial_correlation,
)


def test_rank_biserial_is_extreme_when_all_pairs_favor_left() -> None:
    assert rank_biserial_correlation([1.0, 2.0, 3.0]) == 1.0
    assert rank_biserial_correlation([-1.0, -2.0, -3.0]) == -1.0


def test_zero_paired_differences_have_zero_effect() -> None:
    assert rank_biserial_correlation([0.0, 0.0]) == 0.0
    assert cliffs_delta([1.0, 1.0], [1.0, 1.0]) == 0.0


def test_cliffs_delta_reports_ordered_separation() -> None:
    assert cliffs_delta([3.0, 4.0], [1.0, 2.0]) == 1.0
    assert cliffs_delta([1.0, 2.0], [3.0, 4.0]) == -1.0
