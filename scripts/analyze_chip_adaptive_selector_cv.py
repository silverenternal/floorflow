#!/usr/bin/env python3
"""Leave-one-case-out probe for a lightweight refiner snapshot selector."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/Trinity/src"))

from trinity.floorplan.scoring.total import total_mean

BASE_STEP = 400
FEATURES = (
    "total",
    "structural",
    "overlap",
    "group",
    "mib",
    "boundary",
    "wl",
    "area",
    "trust_region_rms",
    "incremental_rms",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=[800, 1600, 3200])
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0])
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def feature_value(row: dict, name: str) -> float:
    if name == "structural":
        return sum(
            float(row[key])
            for key in ("overlap", "group", "mib", "boundary", "area")
        )
    return float(row[name])


def grouped(report: dict) -> dict[int, dict[int, dict]]:
    observable = {
        (int(row["index"]), int(row["refiner_steps"])): row
        for row in report["rows"]
    }
    legalized = {
        (int(row["index"]), int(row["refiner_steps"])): row
        for row in report["legalized"]
    }
    if set(observable) != set(legalized):
        raise ValueError("observable and legalized rows do not align")
    result: dict[int, dict[int, dict]] = {}
    for key, obs in observable.items():
        index, steps = key
        result.setdefault(index, {})[steps] = {
            "observable": obs,
            "legalized": legalized[key],
        }
    return result


def vector(base: dict, current: dict, blocks: int) -> np.ndarray:
    values = []
    for name in FEATURES:
        before = feature_value(base["observable"], name)
        after = feature_value(current["observable"], name)
        values.append((after - before) / max(abs(before), 1.0e-8))
    values.extend(
        [
            float(blocks) / 120.0,
            feature_value(current["observable"], "total"),
            float(current["observable"]["trust_region_rms"]),
            float(current["observable"]["incremental_rms"]),
        ]
    )
    return np.asarray(values, dtype=np.float64)


def design(cases: dict[int, dict[int, dict]], steps: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_rows = []
    y_rows = []
    groups = []
    for index, case in sorted(cases.items()):
        base = case[BASE_STEP]
        blocks = int(base["legalized"]["blocks"])
        base_cost = float(base["legalized"]["legalized_hard_cost"])
        for step in steps:
            row = case[step]
            x_rows.append(vector(base, row, blocks))
            y_rows.append(
                np.log(
                    float(row["legalized"]["legalized_hard_cost"])
                    / max(base_cost, 1.0e-12)
                )
            )
            groups.append(index)
    return np.stack(x_rows), np.asarray(y_rows), np.asarray(groups)


def fit_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, alpha: float) -> np.ndarray:
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale = np.where(scale < 1.0e-8, 1.0, scale)
    train = (x_train - mean) / scale
    test = (x_test - mean) / scale
    train = np.concatenate([train, np.ones((len(train), 1))], axis=1)
    test = np.concatenate([test, np.ones((len(test), 1))], axis=1)
    penalty = np.eye(train.shape[1]) * alpha
    penalty[-1, -1] = 0.0
    weights = np.linalg.solve(train.T @ train + penalty, train.T @ y_train)
    return test @ weights


def summarize(selected: list[dict]) -> dict:
    costs = [float(row["legalized"]["legalized_hard_cost"]) for row in selected]
    blocks = [int(row["legalized"]["blocks"]) for row in selected]
    return {
        "cases": len(selected),
        "mean_refiner_steps": float(
            np.mean([int(row["refiner_steps"]) for row in selected])
        ),
        "legalized_hard_cost": total_mean(costs, blocks),
        "legalized_v_rel": float(
            np.mean(
                [
                    float(row["legalized"]["legalized_v_rel"])
                    for row in selected
                ]
            )
        ),
        "legalized_feasible_fraction": float(
            np.mean(
                [
                    bool(row["legalized"]["legalized_feasible"])
                    for row in selected
                ]
            )
        ),
        "step_counts": {
            str(step): sum(
                int(row["refiner_steps"]) == step for row in selected
            )
            for step in sorted({int(row["refiner_steps"]) for row in selected})
        },
    }


def fixed(cases: dict[int, dict[int, dict]], step: int) -> dict:
    return summarize(
        [
            {
                "refiner_steps": step,
                **case[step],
            }
            for case in cases.values()
        ]
    )


def main() -> None:
    args = parse_args()
    report = load(args.snapshots)
    cases = grouped(report)
    steps = sorted(set(args.steps))
    if BASE_STEP not in {step for case in cases.values() for step in case}:
        raise ValueError("base step 400 is required")
    x, y, groups = design(cases, steps)
    results = {}
    for alpha in args.alphas:
        selected = []
        for index in sorted(cases):
            train = groups != index
            test_rows = np.flatnonzero(groups == index)
            predicted = fit_predict(x[train], y[train], x[test_rows], alpha)
            best_offset = int(np.argmin(predicted))
            step = steps[best_offset]
            selected.append({"refiner_steps": step, **cases[index][step]})
        results[str(alpha)] = summarize(selected)
    output = {
        "version": "chip-sota-adaptive-selector-cv-v1",
        "status": "leave_one_case_out_diagnostic_only",
        "snapshot_report": str(args.snapshots),
        "case_manifest": report["case_manifest"],
        "cases": len(cases),
        "base_step": BASE_STEP,
        "candidate_steps": steps,
        "fixed_1600": fixed(cases, 1600),
        "selector_by_alpha": results,
        "claim_boundary": (
            "Cross-validation is retrospective and uses cost labels to fit a "
            "selector on other cases. It can only license a later frozen "
            "calibration/validation experiment, never a deployment claim."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output))


if __name__ == "__main__":
    main()
