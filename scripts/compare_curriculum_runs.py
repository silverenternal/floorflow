#!/usr/bin/env python3
"""Compare equal-budget control and curriculum training runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MAXIMIZE = (
    "val_room_macro_iou",
    "val_room_group_macro_iou",
    "val_opening_f1",
    "val_furniture_iou",
    "val_wall_iou",
)
MINIMIZE = ("val_room_partition", "val_loss")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--treatment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_last(path: Path) -> dict:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError(f"empty training log: {path}")
    return records[-1]


def load_records(path: Path) -> list[dict]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError(f"empty training log: {path}")
    return records


def main() -> None:
    args = parse_args()
    control_records = load_records(args.control)
    treatment_records = load_records(args.treatment)
    control = control_records[-1]
    treatment = treatment_records[-1]
    deltas = {}
    improved = 0
    regressed = 0
    for metric in MAXIMIZE:
        delta = float(treatment[metric]) - float(control[metric])
        deltas[metric] = delta
        improved += int(delta > 0)
        regressed += int(delta < 0)
    for metric in MINIMIZE:
        delta = float(treatment[metric]) - float(control[metric])
        deltas[metric] = delta
        improved += int(delta < 0)
        regressed += int(delta > 0)
    report = {
        "status": "pass" if improved > regressed else "negative",
        "equal_compute": {
            "control_seconds": float(control["train_seconds"]),
            "treatment_seconds": float(treatment["train_seconds"]),
            "control_samples_per_second": float(
                control.get("train_samples_per_second", 0.0)
            ),
            "treatment_samples_per_second": float(
                treatment.get("train_samples_per_second", 0.0)
            ),
            "control_steps": int(control.get("global_step", 0)),
            "treatment_steps": int(treatment.get("global_step", 0)),
        },
        "trajectory": {
            "control": [
                {
                    "epoch": row.get("epoch"),
                    "seconds": row.get("train_seconds"),
                    "samples_per_second": row.get(
                        "train_samples_per_second"
                    ),
                    "room_macro_iou": row.get("val_room_macro_iou"),
                    "room_group_macro_iou": row.get(
                        "val_room_group_macro_iou"
                    ),
                }
                for row in control_records
            ],
            "treatment": [
                {
                    "epoch": row.get("epoch"),
                    "seconds": row.get("train_seconds"),
                    "samples_per_second": row.get(
                        "train_samples_per_second"
                    ),
                    "room_macro_iou": row.get("val_room_macro_iou"),
                    "room_group_macro_iou": row.get(
                        "val_room_group_macro_iou"
                    ),
                }
                for row in treatment_records
            ],
        },
        "metrics": {
            metric: {
                "control": float(control[metric]),
                "treatment": float(treatment[metric]),
                "delta": deltas[metric],
            }
            for metric in (*MAXIMIZE, *MINIMIZE)
        },
        "improved_metrics": improved,
        "regressed_metrics": regressed,
        "best_of_n_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
