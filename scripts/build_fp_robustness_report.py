#!/usr/bin/env python3
"""Summarize hard-metric degradation under control corruption."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

METRICS = (
    "fpeval_validity",
    "fpeval_watertight",
    "conditional_watertight",
    "fpeval_edit_cost",
    "control_room_pixel_preservation",
    "room_adjacency_f1",
    "door_contact_room_fraction",
    "door_graph_components",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Corruption rate and metrics JSON, formatted as RATE=PATH.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validity-floor", type=float, default=1.0)
    parser.add_argument("--watertight-floor", type=float, default=0.95)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for value in args.run:
        rate_text, path_text = value.split("=", 1)
        metrics = json.loads(Path(path_text).read_text(encoding="utf-8"))
        rows.append(
            {
                "drop_fraction": float(rate_text),
                "metrics_path": path_text,
                "control_contract": metrics.get("control_contract"),
                **{key: metrics["metrics"][key] for key in METRICS},
            }
        )
    rows.sort(key=lambda row: row["drop_fraction"])
    clean = next(
        (row for row in rows if row["drop_fraction"] == 0.0),
        rows[0],
    )
    for row in rows:
        row["retention_vs_clean"] = {
            "fpeval_validity": (
                row["fpeval_validity"] / clean["fpeval_validity"]
                if clean["fpeval_validity"]
                else None
            ),
            "conditional_watertight": (
                row["conditional_watertight"] / clean["conditional_watertight"]
                if clean["conditional_watertight"]
                else None
            ),
            "room_adjacency_f1": (
                row["room_adjacency_f1"] / clean["room_adjacency_f1"]
                if clean["room_adjacency_f1"]
                else None
            ),
            "edit_cost_multiplier": (
                row["fpeval_edit_cost"] / clean["fpeval_edit_cost"]
                if clean["fpeval_edit_cost"]
                else None
            ),
        }
    mild = next(
        (row for row in rows if row["drop_fraction"] > 0.0),
        None,
    )
    report = {
        "status": "complete",
        "comparison": "spatial block dropout of serialized oracle controls",
        "rows": rows,
        "checks": {
            "clean_validity_floor": clean["fpeval_validity"]
            >= args.validity_floor,
            "clean_conditional_watertight_floor": clean["conditional_watertight"]
            >= args.watertight_floor,
            "mild_validity_floor": (
                mild["fpeval_validity"] >= args.validity_floor
                if mild is not None
                else None
            ),
            "mild_watertight_floor": (
                mild["conditional_watertight"] >= args.watertight_floor
                if mild is not None
                else None
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
