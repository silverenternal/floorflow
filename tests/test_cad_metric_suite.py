from __future__ import annotations

import pytest

from floorflow.evaluation.cad_metric_suite import (
    FAMILIES,
    build_cad_metric_report,
    evaluate_hard_floors,
    metric_agreement,
    pareto_front,
)


def _rows() -> list[dict[str, float]]:
    return [
        {
            "room_macro_iou": 0.10,
            "room_support_iou": 0.75,
            "fpeval_room_polygon_f1": 0.40,
            "instance_geometric_iou": 0.35,
            "fpeval_wall_f1": 0.60,
            "fpeval_opening_f1": 0.30,
            "fpeval_validity": 1.0,
            "fpeval_watertight": 1.0,
            "conditional_watertight": 1.0,
            "room_coverage": 0.92,
            "room_target_recall": 0.91,
            "outside_footprint_pixels": 0.0,
        },
        {
            "room_macro_iou": 0.20,
            "room_support_iou": 0.80,
            "fpeval_room_polygon_f1": 0.50,
            "instance_geometric_iou": 0.45,
            "fpeval_wall_f1": 0.70,
            "fpeval_opening_f1": 0.20,
            "fpeval_validity": 0.0,
            "fpeval_watertight": 0.0,
            "conditional_watertight": 0.0,
            "room_coverage": 0.96,
            "room_target_recall": 0.97,
            "outside_footprint_pixels": 10.0,
        },
    ]


def test_report_separates_families_and_raw_scope() -> None:
    report = build_cad_metric_report(_rows())
    assert report["protocol"]["micro_macro_instance_separated"]
    assert set(report["families"]) == set(FAMILIES)
    assert report["summary"]["room_macro_iou"] == pytest.approx(0.15)
    assert "fpeval_edit_cost" in report["missing_metrics"]


def test_hard_gate_rejects_illegal_high_iou_candidate() -> None:
    report = build_cad_metric_report(_rows())
    assert report["hard_gate"]["passed"] is False
    assert report["hard_gate"]["requirements"]["fpeval_validity"]["passed"] is False
    assert report["hard_gate"]["requirements"]["outside_footprint_pixels"]["passed"] is False


def test_metric_agreement_detects_sign_inversion() -> None:
    agreement = metric_agreement(
        _rows(),
        pairs=(("room_macro_iou", "fpeval_opening_f1"),),
    )
    result = agreement["room_macro_iou__fpeval_opening_f1"]
    assert result["cases"] == 2
    assert result["pearson"] is not None and result["pearson"] < 0.0
    assert result["sign_inversions"] == 2


def test_hard_floor_direction_for_minimize_metric() -> None:
    summary = {"outside_footprint_pixels": 1.0}
    result = evaluate_hard_floors(
        summary,
        {"outside_footprint_pixels": 0.0},
    )
    assert result["outside_footprint_pixels"]["passed"] is False


def test_pareto_rejects_illegal_and_dominated_candidates() -> None:
    candidates = [
        {
            "name": "illegal",
            "hard_gate_passed": False,
            "summary": {
                "room_macro_iou": 0.99,
                "fpeval_room_polygon_f1": 0.99,
                "instance_geometric_iou": 0.99,
            },
        },
        {
            "name": "semantic",
            "hard_gate_passed": True,
            "summary": {
                "room_macro_iou": 0.30,
                "fpeval_room_polygon_f1": 0.20,
                "instance_geometric_iou": 0.50,
            },
        },
        {
            "name": "vector",
            "hard_gate_passed": True,
            "summary": {
                "room_macro_iou": 0.25,
                "fpeval_room_polygon_f1": 0.40,
                "instance_geometric_iou": 0.50,
            },
        },
        {
            "name": "dominated",
            "hard_gate_passed": True,
            "summary": {
                "room_macro_iou": 0.20,
                "fpeval_room_polygon_f1": 0.10,
                "instance_geometric_iou": 0.50,
            },
        },
    ]
    front = pareto_front(candidates)
    assert {candidate["name"] for candidate in front} == {"semantic", "vector"}
