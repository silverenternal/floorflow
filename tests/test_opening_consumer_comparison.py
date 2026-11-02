from __future__ import annotations

import pytest

from floorflow.evaluation.opening_consumer_comparison import (
    compare_opening_reports,
)


def _row(index: int, opening: float) -> dict:
    return {
        "index": index,
        "case": str(index),
        "room_macro_iou": 0.5,
        "room_support_iou": 0.7,
        "instance_instance_f1_at_threshold": 0.4,
        "instance_count_abs_error": 1.0,
        "fpeval_wall_f1": 0.6,
        "fpeval_opening_f1": opening,
        "node_f1": 0.8,
        "edge_overlap": 0.5,
        "ged": 2.0,
        "exact": True,
        "validity": 1.0,
        "functional_watertight": 1.0,
        "wall_continuity_rate": 0.99,
        "outside_footprint_fraction": 0.0,
    }


def _report(rows: list[dict]) -> dict:
    return {
        "target_used_at_inference": False,
        "test_opened": False,
        "cases": {"blend": rows},
        "summary": {
            "blend": {
                "validity_mean": 1.0,
                "functional_watertight_mean": 1.0,
                "outside_footprint_fraction_mean": 0.0,
            }
        },
    }


def test_opening_comparison_uses_common_indices_and_exact_ged() -> None:
    baseline = _report([_row(0, 0.1), _row(1, 0.2), _row(2, 0.3)])
    candidate = _report([_row(0, 0.4), _row(1, 0.5)])
    candidate["cases"]["blend"][1]["exact"] = False
    result = compare_opening_reports(
        baseline,
        candidate,
        iterations=100,
        seed=7,
    )
    assert result["cases"] == 2
    assert result["paired"]["fpeval_opening_f1"]["mean_delta"] == pytest.approx(0.3)
    assert result["paired"]["ged"]["comparison_cases"] == 1
    assert result["gates"]["opening_f1_ci_excludes_zero"] is True
