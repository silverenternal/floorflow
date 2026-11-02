#!/usr/bin/env python3
"""Compare room-program supervision with query-token conditioning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--supervision-only", type=Path, required=True)
    parser.add_argument("--conditioning", type=Path, required=True)
    parser.add_argument("--full-head", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def best_record(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    history = report.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError(f"{path} has no training history")
    row = max(
        history,
        key=lambda value: float(
            value.get("selection_score", value["best_selection_score"])
        ),
    )
    metrics = row["blends"][str(row["best_blend"])]
    return {
        "epoch": int(row["epoch"]),
        "selection_score": float(
            row.get("selection_score", row["best_selection_score"])
        ),
        "best_blend": float(row["best_blend"]),
        "room_iou": float(metrics["room_macro_iou"]),
        "instance_f1": float(metrics["instance_room_f1"]),
        "type_accuracy": float(metrics["instance_type_accuracy"]),
        "count_mae": float(metrics["instance_count_abs_error"]),
        "support_iou": float(metrics["room_support_iou"]),
    }


def main() -> None:
    args = parse_args()
    supervision = best_record(args.supervision_only)
    conditioning = best_record(args.conditioning)
    delta = {
        key: conditioning[key] - supervision[key]
        for key in supervision
    }
    gains = {
        "room_iou": delta["room_iou"] > 0.0,
        "instance_f1": delta["instance_f1"] > 0.0,
        "count_mae": delta["count_mae"] < 0.0,
    }
    report = {
        "schema_version": "room-program-conditioning-screen-v1",
        "decision": (
            "conditioning_candidate"
            if all(gains.values())
            and delta["selection_score"] > 0.0
            else "reject_conditioning"
        ),
        "supervision_only": {
            "report": str(args.supervision_only),
            "best": supervision,
        },
        "conditioning": {
            "report": str(args.conditioning),
            "best": conditioning,
        },
        "conditioning_minus_supervision": delta,
        "joint_quality_gains": gains,
        "claim_boundary": (
            "Matched 64-train/32-validation class-recovery screen; not a "
            "full-data or external SOTA result."
        ),
    }
    if args.full_head is not None:
        full_head = best_record(args.full_head)
        report["full_head_conditioning"] = {
            "report": str(args.full_head),
            "best": full_head,
            "minus_class_only_conditioning": {
                key: full_head[key] - conditioning[key]
                for key in conditioning
            },
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

