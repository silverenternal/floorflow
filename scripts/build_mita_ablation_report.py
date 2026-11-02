#!/usr/bin/env python3
"""Build a fail-closed MiTA-vs-MHA promotion report from raw evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mha-train", type=Path, required=True)
    parser.add_argument("--mita-train", type=Path, required=True)
    parser.add_argument("--mha-metrics", type=Path, required=True)
    parser.add_argument("--mita-metrics", type=Path, required=True)
    parser.add_argument("--mita-routing", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_history(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def best_epoch(history: list[dict]) -> dict:
    return max(
        history,
        key=lambda row: float(row.get("val_partition_room_macro_iou", -1.0)),
    )


def main() -> None:
    args = parse_args()
    mha_history = read_history(args.mha_train)
    mita_history = read_history(args.mita_train)
    mha_best = best_epoch(mha_history)
    mita_best = best_epoch(mita_history)
    mha_metrics = read_json(args.mha_metrics)["metrics"]
    mita_metrics = read_json(args.mita_metrics)["metrics"]
    routing = read_json(args.mita_routing)
    deltas = {
        key: float(mita_metrics[key] - mha_metrics[key])
        for key in mha_metrics
    }
    no_regression = all(value >= -1.0e-6 for value in deltas.values())
    report = {
        "phase": "A1",
        "status": "promoted" if no_regression and routing["passed"] else "rejected",
        "scope": {
            "task": "contour + room program + coarse-layout control",
            "train_prefix": 1024,
            "validation_cases": 191,
            "epochs": 3,
            "same_initialization": "p6_geometry_full_v1/last.pt",
            "best_of_n_used": False,
        },
        "mha": {
            "best_train_epoch": mha_best,
            "validation_metrics": mha_metrics,
        },
        "mita": {
            "best_train_epoch": mita_best,
            "validation_metrics": mita_metrics,
            "routing_health": routing,
        },
        "delta_mita_minus_mha": deltas,
        "promotion_rule": {
            "all_task_metrics_non_regressing": no_regression,
            "routing_health_passed": bool(routing["passed"]),
        },
        "next_gate": "Only promoted MiTA may enter the frozen 503 full-output evaluation.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
