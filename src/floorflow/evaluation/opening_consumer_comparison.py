"""Paired comparison of identical plans with and without opening evidence."""

from __future__ import annotations

from typing import Any

from floorflow.evaluation.paired_bootstrap import paired_delta

KEYS = (
    "room_macro_iou",
    "room_support_iou",
    "instance_instance_f1_at_threshold",
    "instance_count_abs_error",
    "fpeval_wall_f1",
    "fpeval_opening_f1",
    "node_f1",
    "edge_overlap",
    "ged",
    "validity",
    "functional_watertight",
    "wall_continuity_rate",
    "outside_footprint_fraction",
)
LOWER_IS_BETTER = {"instance_count_abs_error", "ged", "outside_footprint_fraction"}


def compare_opening_reports(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    iterations: int = 5000,
    seed: int = 17,
) -> dict[str, Any]:
    for name, report in (("baseline", baseline), ("candidate", candidate)):
        if report.get("target_used_at_inference") is not False:
            raise ValueError(f"{name} target-free contract missing")
        if report.get("test_opened") is not False:
            raise ValueError(f"{name} test was opened")
    baseline_by_index = {
        int(row["index"]): row for row in baseline["cases"]["blend"]
    }
    candidate_by_index = {
        int(row["index"]): row for row in candidate["cases"]["blend"]
    }
    common = sorted(set(baseline_by_index) & set(candidate_by_index))
    if not common:
        raise ValueError("opening comparison has no common cases")
    paired = {}
    for key in KEYS:
        current = [candidate_by_index[index] for index in common]
        reference = [baseline_by_index[index] for index in common]
        if key == "ged":
            exact = {
                index
                for index in common
                if bool(baseline_by_index[index].get("exact"))
                and bool(candidate_by_index[index].get("exact"))
            }
            current = [candidate_by_index[index] for index in sorted(exact)]
            reference = [baseline_by_index[index] for index in sorted(exact)]
        paired[key] = paired_delta(
            current,
            reference,
            key,
            iterations=iterations,
            seed=seed,
            lower_is_better=key in LOWER_IS_BETTER,
        )
        paired[key]["comparison_cases"] = len(current)
    opening = paired["fpeval_opening_f1"]
    candidate_summary = candidate["summary"]["blend"]
    gates = {
        "opening_f1_positive": opening["mean_delta"] > 0.0,
        "opening_f1_ci_excludes_zero": opening["ci_low"] > 0.0,
        "room_iou_non_regression": paired["room_macro_iou"]["mean_delta"] >= 0.0,
        "wall_f1_non_regression": paired["fpeval_wall_f1"]["mean_delta"] >= 0.0,
        "node_f1_non_regression": paired["node_f1"]["mean_delta"] >= 0.0,
        "edge_overlap_non_regression": paired["edge_overlap"]["mean_delta"] >= 0.0,
        "ged_non_regression": paired["ged"]["mean_delta"] <= 0.0,
        "validity_one": candidate_summary["validity_mean"] == 1.0,
        "watertight_at_least_0_99": (
            candidate_summary["functional_watertight_mean"] >= 0.99
        ),
        "outside_footprint_zero": (
            candidate_summary["outside_footprint_fraction_mean"] <= 1.0e-12
        ),
    }
    return {
        "cases": len(common),
        "paired": paired,
        "ged_exact_protocol": {
            "comparison": "intersection_of_both_exact_cases",
            "baseline_exact_rate": float(
                sum(bool(row.get("exact")) for row in baseline["cases"]["blend"])
                / len(baseline["cases"]["blend"])
            ),
            "candidate_exact_rate": float(
                sum(bool(row.get("exact")) for row in candidate["cases"]["blend"])
                / len(candidate["cases"]["blend"])
            ),
        },
        "gates": {**gates, "passed": all(gates.values())},
        "decision": "promote" if all(gates.values()) else "hold",
    }


__all__ = ["KEYS", "LOWER_IS_BETTER", "compare_opening_reports"]
