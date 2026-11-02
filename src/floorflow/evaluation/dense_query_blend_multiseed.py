"""Audit five-seed dense/query blend promotion without pooling seeds."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

PRIMARY_METRICS = ("room_iou", "instance_f1", "count_mae")
LOWER_IS_BETTER = {
    "count_mae",
    "ged",
    "instance_count_abs_error",
    "outside_footprint_fraction",
}
METRIC_ALIASES = {
    "room_iou": "room_iou",
    "instance_f1": "instance_f1",
    "instance_precision": "instance_precision",
    "instance_recall": "instance_recall",
    "type_accuracy": "type_accuracy",
    "count_mae": "count_mae",
    "room_support_iou": "room_support_iou",
}


def _seed_from_path(path: Path) -> int:
    match = re.search(r"(?:^|_)seed(\d+)(?:_|$)", path.stem)
    if match is None:
        raise ValueError(f"cannot recover seed from {path}")
    return int(match.group(1))


def _direction_improved(delta: float, key: str) -> bool:
    return delta < 0.0 if key in LOWER_IS_BETTER else delta > 0.0


def _metric_records(report: dict[str, Any], key: str) -> tuple[float, float, dict]:
    paired_key = METRIC_ALIASES.get(key, key)
    paired = report["paired"][paired_key]
    dense_summary = report["summary"]["dense"]
    blend_summary = report["summary"]["blend"]
    summary_key = (
        key
        if key in dense_summary
        else f"{key}_mean"
    )
    if summary_key not in dense_summary or summary_key not in blend_summary:
        raise KeyError(f"missing summary metric {key}")
    return (
        float(dense_summary[summary_key]),
        float(blend_summary[summary_key]),
        paired,
    )


def aggregate_split(
    reports: list[dict[str, Any]],
    *,
    metric_keys: tuple[str, ...],
) -> dict[str, Any]:
    """Aggregate seed-level effects while retaining every seed's direction."""

    metrics = {}
    for key in metric_keys:
        records = []
        for report in reports:
            dense, blend, paired = _metric_records(report, key)
            delta = float(paired["mean_delta"])
            records.append(
                {
                    "seed": int(report["seed"]),
                    "path": str(report["path"]),
                    "dense": dense,
                    "blend": blend,
                    "delta": delta,
                    "ci_low": float(paired["ci_low"]),
                    "ci_high": float(paired["ci_high"]),
                    "ci_excludes_zero": bool(
                        float(paired["ci_low"]) > 0.0
                        or float(paired["ci_high"]) < 0.0
                    ),
                    "improved": _direction_improved(delta, key),
                }
            )
        deltas = np.asarray([record["delta"] for record in records])
        ordered = sorted(records, key=lambda record: record["delta"])
        best = ordered[0] if key in LOWER_IS_BETTER else ordered[-1]
        worst = ordered[-1] if key in LOWER_IS_BETTER else ordered[0]
        metrics[key] = {
            "lower_is_better": key in LOWER_IS_BETTER,
            "seed_records": records,
            "delta_mean": float(deltas.mean()),
            "delta_std": float(deltas.std(ddof=1)),
            "delta_min": float(deltas.min()),
            "delta_max": float(deltas.max()),
            "improved_seeds": int(sum(record["improved"] for record in records)),
            "non_regression_seeds": int(
                sum(
                    record["delta"] <= 0.0
                    if key in LOWER_IS_BETTER
                    else record["delta"] >= 0.0
                    for record in records
                )
            ),
            "ci_excludes_zero_seeds": int(
                sum(record["ci_excludes_zero"] for record in records)
            ),
            "best_seed": int(best["seed"]),
            "worst_seed": int(worst["seed"]),
        }
    return {
        "seeds": sorted(int(report["seed"]) for report in reports),
        "cases_per_seed": sorted(
            {int(report["cases"]) for report in reports}
        ),
        "metrics": metrics,
        "paths": [str(report["path"]) for report in reports],
    }


def load_split_reports(paths: list[Path]) -> list[dict[str, Any]]:
    import json

    reports = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("target_used_at_inference") is not False:
            raise ValueError(f"target-free contract missing in {path}")
        if payload.get("test_opened") not in {None, False}:
            raise ValueError(f"test was opened in {path}")
        reports.append(
            {
                **payload,
                "seed": _seed_from_path(path),
                "path": path,
                "cases": int(payload["validation_cases"]),
            }
        )
    seeds = [report["seed"] for report in reports]
    if len(seeds) != len(set(seeds)):
        raise ValueError("duplicate seed reports")
    return sorted(reports, key=lambda report: report["seed"])


def promotion_gates(
    splits: dict[str, dict[str, Any]],
    *,
    expected_seeds: list[int],
) -> dict[str, Any]:
    required_splits = {"val491", "cal64", "holdout128", "structure_val491"}
    missing = sorted(required_splits.difference(splits))
    coverage = {}
    for name, split in splits.items():
        coverage[name] = split["seeds"] == sorted(expected_seeds)
    checks = {
        "all_required_splits_present": not missing,
        "all_splits_cover_five_seeds": bool(coverage)
        and all(coverage.values()),
    }
    if checks["all_required_splits_present"] and checks["all_splits_cover_five_seeds"]:
        val = splits["val491"]["metrics"]
        checks["val_primary_four_of_five"] = all(
            val[key]["improved_seeds"] >= 4 for key in PRIMARY_METRICS
        )
        checks["val_support_five_of_five"] = (
            val["room_support_iou"]["non_regression_seeds"] == 5
        )
        checks["split_direction_four_of_five"] = all(
            splits[split]["metrics"][key]["improved_seeds"] >= 4
            for split in ("cal64", "holdout128")
            for key in PRIMARY_METRICS
        )
        structure = splits["structure_val491"]["metrics"]
        checks["structure_room_iou_non_regression"] = (
            structure["room_macro_iou"]["non_regression_seeds"] == 5
        )
        checks["structure_validity_one"] = (
            structure["validity"]["seed_records"]
            and all(
                record["blend"] == 1.0
                for record in structure["validity"]["seed_records"]
            )
        )
        checks["structure_watertight_at_least_0_99"] = all(
            record["blend"] >= 0.99
            for record in structure["functional_watertight"]["seed_records"]
        )
        checks["structure_outside_zero"] = all(
            record["blend"] <= 1.0e-12
            for record in structure["outside_footprint_fraction"]["seed_records"]
        )
    return {
        "missing_splits": missing,
        "coverage": coverage,
        "checks": checks,
        "passed": bool(checks) and all(checks.values()),
    }


__all__ = [
    "LOWER_IS_BETTER",
    "PRIMARY_METRICS",
    "aggregate_split",
    "load_split_reports",
    "promotion_gates",
]
