"""Matched comparison for dense and query furniture arms."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def _mean(values: Sequence[float]) -> float:
    return sum(float(value) for value in values) / max(1, len(values))


def _final_three_mean(history: Sequence[Mapping[str, Any]], key: str) -> float:
    if not history:
        return 0.0
    return _mean([float(row[key]) for row in history[-3:]])


def _trend(history: Sequence[Mapping[str, Any]], key: str) -> float:
    if len(history) < 2:
        return 0.0
    return float(history[-1][key]) - float(history[-3][key])


def compare_furniture_arm_reports(
    dense: Mapping[str, Any],
    query: Mapping[str, Any],
    *,
    safety_tolerance: float = 0.005,
) -> dict[str, Any]:
    """Compare final-three-checkpoint quality and safety regressions."""

    metric_names = (
        "furniture_f1",
        "room_macro_iou",
        "instance_f1",
        "wall_f1",
        "opening_f1",
    )
    rows = {}
    for name in metric_names:
        dense_value = _final_three_mean(dense["history"], name)
        query_value = _final_three_mean(query["history"], name)
        rows[name] = {
            "dense_final3_mean": dense_value,
            "query_final3_mean": query_value,
            "query_minus_dense": query_value - dense_value,
            "dense_last_minus_third_last": _trend(
                dense["history"],
                name,
            ),
            "query_last_minus_third_last": _trend(
                query["history"],
                name,
            ),
        }
    safety = (
        "room_macro_iou",
        "instance_f1",
        "wall_f1",
        "opening_f1",
    )
    safety_pass = all(
        rows[name]["query_minus_dense"] >= -float(safety_tolerance)
        for name in safety
    )
    furniture_positive = rows["furniture_f1"]["query_minus_dense"] > 0.0
    furniture_trend_positive = (
        rows["furniture_f1"]["query_last_minus_third_last"] > 0.0
    )
    furniture_pass = furniture_positive and furniture_trend_positive
    return {
        "schema_version": "cad-furniture-ablation-comparison-v1",
        "metrics": rows,
        "gates": {
            "safety_tolerance": float(safety_tolerance),
            "safety_pass": safety_pass,
            "furniture_positive": furniture_positive,
            "furniture_trend_positive": furniture_trend_positive,
            "furniture_pass": furniture_pass,
            "promotion_ready": safety_pass and furniture_pass,
        },
        "decision": (
            "advance_to_matched_capacity_and_second_seed"
            if safety_pass and furniture_pass
            else "keep_query_disabled_and_diagnose_representation_or_budget"
        ),
    }


__all__ = ["compare_furniture_arm_reports"]
