#!/usr/bin/env python3
"""Fail-closed selection across val-only overlay candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

GUARDS = {
    "room_macro_iou": 0.002,
    "instance_geometric_iou": 0.005,
    "instance_type_accuracy": 0.01,
}
GAINS = (
    "room_macro_iou",
    "instance_geometric_iou",
    "instance_type_accuracy",
    "opening_f1",
    "furniture_iou",
    "wall_iou",
    "fpeval_wall_f1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        type=lambda value: tuple(value.split("=", 1)),
        default=[],
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def flatten(report: dict) -> dict[str, float]:
    room = report["room_metrics"]["overlay"]
    objects = report["object_metrics"]["overlay"]
    summary = report["cad_metrics"]["summary"]
    return {
        "room_macro_iou": float(room["room_macro_iou"]),
        "instance_geometric_iou": float(room["instance_geometric_iou"]),
        "instance_type_accuracy": float(room["instance_type_accuracy"]),
        "opening_f1": float(objects["opening_f1"]),
        "furniture_iou": float(objects["furniture_iou"]),
        "wall_iou": float(objects["wall_iou"]),
        "fpeval_wall_f1": float(room["fpeval_wall_f1"]),
        "fpeval_opening_f1": float(room["fpeval_opening_f1"]),
        "fpeval_room_polygon_f1": float(room["fpeval_room_polygon_f1"]),
        "unconditional_watertight": float(room["fpeval_watertight"]),
        "conditional_watertight": float(summary["conditional_watertight"]),
        "fpeval_validity": float(summary["fpeval_validity"]),
        "room_target_recall": float(summary["room_target_recall"]),
        "outside_footprint_pixels": float(summary["outside_footprint_pixels"]),
        "balanced_cad": float(report["cad_metrics"]["balanced_secondary"]["score"]),
    }


def decision(reference: dict, candidate: dict) -> dict[str, object]:
    reasons = []
    improvements = []
    hard_floor = (
        candidate["fpeval_validity"] >= 1.0
        and candidate["conditional_watertight"] >= 0.95
        and candidate["room_target_recall"] >= 0.95
        and candidate["outside_footprint_pixels"] == 0.0
    )
    if not hard_floor:
        reasons.append("hard_floor")
    for name, tolerance in GUARDS.items():
        if candidate[name] < reference[name] - tolerance:
            reasons.append(f"regressed_{name}")
    for name in GAINS:
        if candidate[name] > reference[name] + 0.002:
            improvements.append(name)
    if not improvements:
        reasons.append("no_material_gain")
    return {
        "eligible": not reasons,
        "reasons": reasons,
        "improvements": improvements,
        "hard_floor": hard_floor,
        "balanced_delta": candidate["balanced_cad"]
        - reference["balanced_cad"],
    }


def main() -> None:
    args = parse_args()
    reference = flatten(load(args.reference))
    rows = []
    for name, path in args.candidate:
        candidate = flatten(load(Path(path)))
        rows.append(
            {
                "name": name,
                "report": path,
                "metrics": candidate,
                **decision(reference, candidate),
            }
        )
    eligible = [row for row in rows if row["eligible"]]
    selected = max(
        eligible,
        key=lambda row: row["metrics"]["balanced_cad"],
        default=None,
    )
    report = {
        "selection_split": "val",
        "best_of_n_used": False,
        "test_used_for_selection": False,
        "reference": {"report": str(args.reference), "metrics": reference},
        "candidates": rows,
        "selected": selected,
        "rollback_to_reference": selected is None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
