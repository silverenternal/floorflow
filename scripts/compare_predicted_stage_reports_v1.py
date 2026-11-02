#!/usr/bin/env python3
"""Compare two predicted-control stage reports by shared case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from floorflow.evaluation.paired_bootstrap import paired_delta

METRICS = (
    "fpeval_opening_f1",
    "fpeval_wall_f1",
    "fpeval_room_f1",
    "room_macro_iou",
    "node_f1",
    "edge_overlap",
    "ged",
    "validity",
    "functional_watertight",
    "wall_continuity_rate",
    "outside_footprint_fraction",
)
LOWER_IS_BETTER = {"ged", "outside_footprint_fraction"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))["cases"]
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))["cases"]
    baseline_by_case = {row["case"]: row for row in baseline}
    rows = [
        (row, baseline_by_case[row["case"]])
        for row in candidate
        if row["case"] in baseline_by_case
    ]
    if not rows:
        raise ValueError("stage reports have no shared cases")
    paired = {}
    for metric in METRICS:
        if metric not in rows[0][0] or metric not in rows[0][1]:
            continue
        paired[metric] = paired_delta(
            [{"value": row[metric]} for row, _base in rows],
            [{"value": base[metric]} for _row, base in rows],
            "value",
            iterations=args.bootstrap,
            seed=17,
            lower_is_better=metric in LOWER_IS_BETTER,
        )
    gate = {
        "opening_ci_lower_positive": paired["fpeval_opening_f1"]["ci_low"]
        > 0.0,
        "wall_nonregression": paired["fpeval_wall_f1"]["mean_delta"] >= 0.0,
        "room_nonregression": paired["fpeval_room_f1"]["mean_delta"] >= 0.0,
        "node_nonregression": paired["node_f1"]["mean_delta"] >= 0.0,
        "edge_nonregression": paired["edge_overlap"]["mean_delta"] >= 0.0,
        "ged_nonregression": paired["ged"]["mean_delta"] <= 0.0,
        "validity_preserved": paired["validity"]["mean_delta"] == 0.0,
        "watertight_preserved": (
            paired["functional_watertight"]["mean_delta"] == 0.0
        ),
        "outside_footprint_preserved": (
            paired["outside_footprint_fraction"]["mean_delta"] == 0.0
        ),
    }
    gate["promoted"] = all(gate.values())
    report = {
        "schema_version": "predicted-stage-paired-comparison-v1",
        "candidate": str(args.candidate),
        "baseline": str(args.baseline),
        "cases": len(rows),
        "paired": paired,
        "gate": gate,
        "decision": "promote_opening_detector" if gate["promoted"] else "hold",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"paired": paired, "gate": gate}, indent=2))


if __name__ == "__main__":
    main()
