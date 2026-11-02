#!/usr/bin/env python3
"""Freeze and validate a bounded adaptive-refiner stopping policy."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/Trinity/src"))

from trinity.floorplan.scoring.total import total_mean

DEFAULT_THRESHOLDS = (
    0.0,
    0.001,
    0.0025,
    0.005,
    0.01,
    0.02,
    0.05,
    0.1,
    0.15,
    0.2,
    0.3,
    0.5,
)
ENERGY_METRICS = (
    "total",
    "structural",
    "overlap",
    "group",
    "mib",
    "boundary",
    "wl",
    "area",
)
RANK_METRICS = (*ENERGY_METRICS, "trust_region_rms", "incremental_rms")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--mode", choices=("calibrate", "validate"), required=True)
    parser.add_argument("--policy", type=Path, default=None)
    parser.add_argument("--control-steps", type=int, default=1600)
    parser.add_argument("--thresholds", type=float, nargs="+", default=list(DEFAULT_THRESHOLDS))
    parser.add_argument("--metrics", nargs="+", choices=ENERGY_METRICS, default=list(ENERGY_METRICS))
    parser.add_argument("--min-steps", type=int, default=800)
    parser.add_argument("--tolerance", type=float, default=0.005)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1.0e-15:
        return 1.0 if abs(numerator) < 1.0e-15 else float("inf")
    return numerator / denominator


def index_rows(rows: list[dict], key: str) -> dict[int, dict[int, dict]]:
    grouped: dict[int, dict[int, dict]] = {}
    for row in rows:
        case_index = int(row["index"])
        steps = int(row["refiner_steps"])
        case = grouped.setdefault(case_index, {})
        if steps in case:
            raise ValueError(f"duplicate {key} row for case {case_index} step {steps}")
        case[steps] = row
    return grouped


def validate_snapshot_report(report: dict) -> tuple[dict[int, dict[int, dict]], dict[int, dict[int, dict]], list[int]]:
    snapshots = sorted({int(step) for step in report["snapshots"]})
    observables = index_rows(report["rows"], "observable")
    legalized = index_rows(report["legalized"], "legalized")
    if set(observables) != set(legalized):
        raise ValueError("observable and legalized reports cover different cases")
    for case_index, case in observables.items():
        if set(case) != set(snapshots) or set(legalized[case_index]) != set(snapshots):
            raise ValueError(f"case {case_index} does not cover every frozen snapshot")
    return observables, legalized, snapshots


def metric_value(row: dict, name: str) -> float:
    if name == "structural":
        return sum(
            float(row[key])
            for key in ("overlap", "group", "mib", "boundary", "area")
        )
    return float(row[name])


def apply_policy(
    observables: dict[int, dict[int, dict]],
    snapshots: list[int],
    policy: dict,
) -> dict[int, int]:
    kind = policy["kind"]
    if kind == "fixed":
        steps = int(policy["steps"])
        return {case_index: steps for case_index in observables}
    if kind == "observable_rank":
        metric = policy["metric"]
        direction = policy["direction"]
        sign = 1.0 if direction == "min" else -1.0
        selected = {}
        for case_index, case in observables.items():
            selected[case_index] = min(
                snapshots,
                key=lambda step: (sign * metric_value(case[step], metric), step),
            )
        return selected
    if kind != "energy_plateau":
        raise ValueError(f"unsupported policy kind: {kind}")

    metric = policy["metric"]
    threshold = float(policy["threshold"])
    min_steps = int(policy["min_steps"])
    eligible = [step for step in snapshots if step >= min_steps]
    if not eligible:
        raise ValueError("no snapshot at or above policy min_steps")
    selected = {}
    for case_index, case in observables.items():
        choice = eligible[-1]
        for previous_steps, steps in itertools.pairwise(eligible):
            before = metric_value(case[previous_steps], metric)
            after = metric_value(case[steps], metric)
            relative_change = abs(after - before) / max(abs(before), 1.0e-12)
            if relative_change <= threshold:
                choice = steps
                break
        selected[case_index] = choice
    return selected


def evaluate_selection(
    selected: dict[int, int],
    legalized: dict[int, dict[int, dict]],
) -> dict:
    rows = [
        legalized[case_index][steps]
        for case_index, steps in sorted(selected.items())
    ]
    costs = [float(row["legalized_hard_cost"]) for row in rows]
    blocks = [int(row["blocks"]) for row in rows]
    return {
        "cases": len(rows),
        "mean_refiner_steps": float(np.mean(list(selected.values()))),
        "legalized_hard_cost": total_mean(costs, blocks),
        "legalized_v_rel": float(
            np.mean([float(row["legalized_v_rel"]) for row in rows])
        ),
        "legalized_feasible_fraction": float(
            np.mean([bool(row["legalized_feasible"]) for row in rows])
        ),
        "selected_steps": {
            str(index): steps for index, steps in sorted(selected.items())
        },
    }


def policy_row(selected: dict[int, int], result: dict, control: dict, tolerance: float) -> dict:
    cost_ratio = safe_ratio(
        result["legalized_hard_cost"],
        control["legalized_hard_cost"],
    )
    v_rel_ratio = safe_ratio(
        result["legalized_v_rel"],
        control["legalized_v_rel"],
    )
    return {
        "mean_refiner_steps": result["mean_refiner_steps"],
        "legalized_hard_cost": result["legalized_hard_cost"],
        "legalized_v_rel": result["legalized_v_rel"],
        "legalized_feasible_fraction": result["legalized_feasible_fraction"],
        "cost_ratio_vs_control": cost_ratio,
        "v_rel_ratio_vs_control": v_rel_ratio,
        "passes_metric_guard": (
            cost_ratio <= 1.0 + tolerance
            and v_rel_ratio <= 1.0 + tolerance
            and result["legalized_feasible_fraction"] == 1.0
        ),
        "selected_steps": result["selected_steps"],
    }


def candidate_policies(
    thresholds: list[float],
    metrics: list[str],
    min_steps: int,
    max_steps: int,
) -> list[dict]:
    policies = [
        {"kind": "fixed", "steps": steps, "id": f"fixed-{steps}"}
        for steps in (400, 800, 1600, 3200)
        if steps <= max_steps
    ]
    policies.extend(
        {
            "kind": "energy_plateau",
            "metric": metric,
            "threshold": threshold,
            "min_steps": min_steps,
            "id": f"plateau-{metric}-{threshold:g}",
        }
        for metric in metrics
        for threshold in thresholds
    )
    policies.extend(
        {
            "kind": "observable_rank",
            "metric": metric,
            "direction": direction,
            "id": f"rank-{direction}-{metric}",
        }
        for metric in metrics
        for direction in ("min", "max")
        if metric in RANK_METRICS
    )
    return policies


def calibrate(
    observables: dict[int, dict[int, dict]],
    legalized: dict[int, dict[int, dict]],
    snapshots: list[int],
    control_steps: int,
    thresholds: list[float],
    metrics: list[str],
    min_steps: int,
    tolerance: float,
) -> dict:
    control_selection = {case_index: control_steps for case_index in observables}
    control = evaluate_selection(control_selection, legalized)
    candidates = []
    for policy in candidate_policies(thresholds, metrics, min_steps, snapshots[-1]):
        selected = apply_policy(observables, snapshots, policy)
        result = evaluate_selection(selected, legalized)
        candidates.append(
            {
                "policy": policy,
                **policy_row(selected, result, control, tolerance),
            }
        )
    eligible = [row for row in candidates if row["passes_metric_guard"]]
    if not eligible:
        selected_policy = {"kind": "fixed", "steps": control_steps, "id": f"fixed-{control_steps}"}
        status = "fallback_to_control_no_policy_passed"
    else:
        best = min(
            eligible,
            key=lambda row: (
                row["mean_refiner_steps"],
                row["legalized_hard_cost"],
                row["legalized_v_rel"],
            ),
        )
        selected_policy = best["policy"]
        status = "policy_frozen"
    selected = apply_policy(observables, snapshots, selected_policy)
    evaluation = evaluate_selection(selected, legalized)
    return {
        "status": status,
        "control": {"policy": {"kind": "fixed", "steps": control_steps}, **control},
        "policy": selected_policy,
        "selected": evaluation,
        "thresholds": thresholds,
        "metrics": metrics,
        "min_steps": min_steps,
        "tolerance": tolerance,
        "candidates": candidates,
    }


def main() -> None:
    args = parse_args()
    report = load(args.snapshots)
    observables, legalized, snapshots = validate_snapshot_report(report)
    if args.mode == "calibrate":
        result = calibrate(
            observables,
            legalized,
            snapshots,
            args.control_steps,
            args.thresholds,
            args.metrics,
            args.min_steps,
            args.tolerance,
        )
    else:
        if args.policy is None:
            raise ValueError("--policy is required in validate mode")
        policy_payload = load(args.policy)
        policy = policy_payload["policy"]
        selected = apply_policy(observables, snapshots, policy)
        evaluation = evaluate_selection(selected, legalized)
        control_selection = {case_index: args.control_steps for case_index in observables}
        control = evaluate_selection(control_selection, legalized)
        comparison = policy_row(selected, evaluation, control, args.tolerance)
        compute_win = evaluation["mean_refiner_steps"] < args.control_steps
        equal_budget_strict_win = (
            evaluation["mean_refiner_steps"] <= args.control_steps
            and comparison["legalized_hard_cost"] < control["legalized_hard_cost"]
            and comparison["legalized_v_rel"] < control["legalized_v_rel"]
        )
        result = {
            "status": (
                "validated_compute_win"
                if comparison["passes_metric_guard"] and compute_win
                else "validated_equal_budget_win"
                if comparison["passes_metric_guard"] and equal_budget_strict_win
                else "validation_rejected"
            ),
            "policy": policy,
            "control": control,
            "selected": evaluation,
            "comparison": comparison,
        }
    result.update(
        {
            "mode": args.mode,
            "snapshot_report": str(args.snapshots),
            "snapshot_report_sha256": sha256_file(args.snapshots),
            "snapshot_manifest": report["case_manifest"],
            "target_used_for_stop": False,
            "target_labels_used_for_calibration": args.mode == "calibrate",
            "case_identity_used_for_stop": False,
            "validation_reused_for_fit": False,
            "policy_file_sha256": (
                sha256_file(args.policy) if args.policy is not None else None
            ),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "candidates"}))


if __name__ == "__main__":
    main()
