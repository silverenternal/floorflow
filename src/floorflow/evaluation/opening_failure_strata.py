"""Join opening detector diagnostics with final-consumer outcomes."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from floorflow.evaluation.paired_bootstrap import paired_delta


def _normalize_case(case: str) -> str:
    return case.replace("/", "__", 1)


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def diagnosis_category(row: dict[str, Any]) -> str:
    target = int(row["target_openings"])
    predicted = int(row["predicted_openings"])
    matched = int(row["matched_openings"])
    if target == 0:
        return "false_positive_on_empty" if predicted else "empty_correct"
    missed = max(0, target - matched)
    extra = max(0, predicted - matched)
    if matched == 0:
        return "complete_miss"
    if missed and extra:
        return "mixed_miss_and_extra"
    if missed:
        return "under_detection"
    if extra:
        return "over_detection"
    return "count_correct_residual"


def target_count_bin(target_openings: int) -> str:
    if target_openings == 0:
        return "0"
    if target_openings <= 3:
        return "1-3"
    if target_openings <= 7:
        return "4-7"
    if target_openings <= 11:
        return "8-11"
    return "12+"


def _paired_summary(items: list[dict[str, Any]], iterations: int) -> dict[str, Any]:
    current = [{"value": item["candidate_f1"]} for item in items]
    baseline = [{"value": item["baseline_f1"]} for item in items]
    paired = paired_delta(
        current,
        baseline,
        "value",
        iterations=iterations,
        seed=17,
    )
    losses = [
        max(0.0, item["baseline_f1"] - item["candidate_f1"])
        for item in items
    ]
    gains = [
        max(0.0, item["candidate_f1"] - item["baseline_f1"])
        for item in items
    ]
    return {
        "cases": len(items),
        "baseline_opening_f1": _mean(
            [float(item["baseline_f1"]) for item in items]
        ),
        "candidate_opening_f1": _mean(
            [float(item["candidate_f1"]) for item in items]
        ),
        "paired_delta": paired,
        "worsened_cases": sum(
            item["candidate_f1"] < item["baseline_f1"] for item in items
        ),
        "improved_cases": sum(
            item["candidate_f1"] > item["baseline_f1"] for item in items
        ),
        "f1_loss_sum": float(sum(losses)),
        "f1_gain_sum": float(sum(gains)),
        "oracle_pixel_f1": _mean([float(item["pixel_f1"]) for item in items]),
        "oracle_vector_f1": _mean([float(item["vector_f1"]) for item in items]),
        "oracle_vector_precision": _mean(
            [float(item["vector_precision"]) for item in items]
        ),
        "oracle_vector_recall": _mean(
            [float(item["vector_recall"]) for item in items]
        ),
        "target_openings": _mean(
            [float(item["target_openings"]) for item in items]
        ),
        "predicted_openings": _mean(
            [float(item["predicted_openings"]) for item in items]
        ),
        "matched_openings": _mean(
            [float(item["matched_openings"]) for item in items]
        ),
    }


def _group_summaries(
    items: list[dict[str, Any]],
    group_key: str,
    iterations: int,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        grouped[str(item[group_key])].append(item)
    return {
        key: _paired_summary(grouped[key], iterations)
        for key in sorted(grouped)
    }


def analyze_opening_failure_strata(
    baseline_report: dict[str, Any],
    candidate_report: dict[str, Any],
    oracle_report: dict[str, Any],
    *,
    iterations: int = 5000,
    top_failures: int = 30,
) -> dict[str, Any]:
    baseline = {row["case"]: row for row in baseline_report["cases"]}
    candidate = {row["case"]: row for row in candidate_report["cases"]}
    oracle = {
        _normalize_case(row["case"]): row for row in oracle_report["cases"]
    }
    common = sorted(set(baseline) & set(candidate) & set(oracle))
    if not common:
        raise ValueError("opening reports have no shared cases")
    items = []
    for case in common:
        oracle_row = oracle[case]
        target = int(oracle_row["target_openings"])
        items.append(
            {
                "case": case,
                "baseline_f1": float(baseline[case]["fpeval_opening_f1"]),
                "candidate_f1": float(candidate[case]["fpeval_opening_f1"]),
                "diagnosis": diagnosis_category(oracle_row),
                "target_bin": target_count_bin(target),
                **oracle_row,
            }
        )
    failure_items = [
        item
        for item in items
        if item["candidate_f1"] < item["baseline_f1"]
    ]
    failure_items.sort(
        key=lambda item: (
            item["candidate_f1"],
            item["baseline_f1"] - item["candidate_f1"],
            item["case"],
        )
    )
    overall = _paired_summary(items, iterations)
    total_loss = float(sum(
        max(0.0, item["baseline_f1"] - item["candidate_f1"])
        for item in items
    ))
    by_diagnosis = _group_summaries(items, "diagnosis", iterations)
    for summary in by_diagnosis.values():
        summary["loss_share"] = (
            summary["f1_loss_sum"] / total_loss if total_loss else 0.0
        )
    return {
        "schema_version": "opening-failure-strata-v1",
        "scope": "multiscale detector joined to matched final-consumer plans",
        "target_used_at_inference": False,
        "test_opened": False,
        "cases": len(items),
        "overall": overall,
        "total_f1_loss": total_loss,
        "by_diagnosis": by_diagnosis,
        "by_target_count": _group_summaries(
            items,
            "target_bin",
            iterations,
        ),
        "top_worsened_cases": failure_items[:top_failures],
    }


__all__ = [
    "analyze_opening_failure_strata",
    "diagnosis_category",
    "target_count_bin",
]
