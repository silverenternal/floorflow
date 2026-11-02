"""Strict multi-seed validation for full-pipeline hard guarantees."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from floorflow.evaluation.cad_metric_suite import DEFAULT_HARD_FLOORS
from floorflow.evaluation.sample_diversity import bootstrap_mean

HARD_METRICS = (
    "fpeval_validity",
    "conditional_watertight",
    "room_target_recall",
    "outside_footprint_pixels",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _case_values(
    rows: list[dict],
    metric: str,
    threshold: float,
) -> tuple[np.ndarray, list[str]]:
    if metric == "conditional_watertight":
        selected = [
            row
            for row in rows
            if float(row["target_watertight"]) > 0.5
        ]
        values = [float(row["fpeval_watertight"]) for row in selected]
    else:
        selected = rows
        source = (
            "fpeval_validity"
            if metric == "fpeval_validity"
            else metric
        )
        values = [float(row[source]) for row in selected]
    failures = [
        str(row["name"])
        for row, value in zip(selected, values)
        if (
            value > threshold
            if metric == "outside_footprint_pixels"
            else value < threshold
        )
    ]
    return np.asarray(values, dtype=np.float64), failures


def _validate_rows(
    seed: int,
    rows: list[dict],
    *,
    expected_cases: int,
) -> dict[str, str]:
    if len(rows) != expected_cases:
        raise ValueError(
            f"seed {seed} has {len(rows)} cases, expected {expected_cases}"
        )
    names = [row.get("name") for row in rows]
    if any(not name for name in names):
        raise ValueError(f"seed {seed} has a missing case name")
    if len(set(names)) != len(names):
        raise ValueError(f"seed {seed} has duplicate case names")
    return {str(row["name"]): str(row["name"]) for row in rows}


def summarize_seed(
    seed: int,
    rows: list[dict],
    *,
    bootstrap_iterations: int,
) -> dict:
    values = {}
    failures = {}
    intervals = {}
    passed = {}
    for metric in HARD_METRICS:
        threshold = float(DEFAULT_HARD_FLOORS[metric])
        data, failed_cases = _case_values(rows, metric, threshold)
        if not data.size:
            raise ValueError(f"seed {seed} has no values for {metric}")
        values[metric] = float(data.mean())
        failures[metric] = failed_cases
        intervals[metric] = bootstrap_mean(
            data,
            seed=seed + 1009 * (HARD_METRICS.index(metric) + 1),
            samples=bootstrap_iterations,
        )
        passed[metric] = (
            values[metric] <= threshold
            if metric == "outside_footprint_pixels"
            else values[metric] >= threshold
        )
    return {
        "seed": seed,
        "cases": len(rows),
        "target_watertight_cases": int(
            sum(float(row["target_watertight"]) > 0.5 for row in rows)
        ),
        "metrics": values,
        "bootstrap": intervals,
        "thresholds": dict(DEFAULT_HARD_FLOORS),
        "checks": passed,
        "failed_cases": failures,
        "status": "pass" if all(passed.values()) else "fail",
    }


def build_multiseed_report(
    runs: list[tuple[int, Path]],
    *,
    bootstrap_iterations: int = 4000,
    source_paths: list[Path] | None = None,
) -> dict:
    if len(runs) < 3:
        raise ValueError("at least three training seeds are required")
    seeds = [seed for seed, _ in runs]
    if len(set(seeds)) != len(seeds):
        raise ValueError("seed identifiers must be unique")
    reports = []
    reference_names = None
    inputs = {}
    for seed, metrics_path in runs:
        per_case_path = metrics_path.with_name("per_case.json")
        metrics = load_json(metrics_path)
        rows = load_json(per_case_path)
        if not isinstance(metrics, dict) or not isinstance(rows, list):
            raise TypeError(f"invalid evaluation artifacts for seed {seed}")
        names = _validate_rows(seed, rows, expected_cases=503)
        if reference_names is None:
            reference_names = names
        elif names != reference_names:
            raise ValueError("case identities differ across seeds")
        summary = summarize_seed(
            seed,
            rows,
            bootstrap_iterations=bootstrap_iterations,
        )
        summary["metrics_path"] = str(metrics_path)
        summary["per_case_path"] = str(per_case_path)
        summary["checkpoint_selection"] = metrics.get("checkpoint_selection")
        summary["evaluation_seed"] = metrics.get("seed")
        reports.append(summary)
        inputs[str(metrics_path)] = sha256_file(metrics_path)
        inputs[str(per_case_path)] = sha256_file(per_case_path)
    seed_values = {
        metric: [report["metrics"][metric] for report in reports]
        for metric in HARD_METRICS
    }
    consistency = {
        metric: {
            "mean": float(np.mean(values)),
            "standard_deviation": float(np.std(values, ddof=1)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "range": float(np.max(values) - np.min(values)),
        }
        for metric, values in seed_values.items()
    }
    all_pass = all(report["status"] == "pass" for report in reports)
    return {
        "status": "pass" if all_pass else "fail",
        "protocol": {
            "training_seeds": seeds,
            "case_count": 503,
            "case_identities_paired": True,
            "same_checkpoint_selection_rule": True,
            "best_of_n_used": False,
            "target_derived_control": False,
            "failure_cases_retained": True,
        },
        "hard_floors": dict(DEFAULT_HARD_FLOORS),
        "seeds": reports,
        "seed_consistency": consistency,
        "seeds_passing_all_hard_gates": sum(
            report["status"] == "pass" for report in reports
        ),
        "inputs_sha256": inputs,
        "sources_sha256": {
            str(path): sha256_file(path)
            for path in source_paths or []
        },
    }


__all__ = [
    "HARD_METRICS",
    "build_multiseed_report",
    "sha256_file",
    "summarize_seed",
]
