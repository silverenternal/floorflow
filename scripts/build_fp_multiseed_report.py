#!/usr/bin/env python3
"""Summarize hard-metric replication across completion training seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from floorflow.evaluation.paired_bootstrap import paired_delta

METRICS = (
    "fpeval_validity",
    "fpeval_watertight",
    "conditional_watertight",
    "fpeval_edit_cost",
    "edit_cost_reduction",
    "control_room_pixel_preservation",
    "room_adjacency_f1",
    "door_contact_room_fraction",
    "door_graph_components",
    "room_support_iou",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="SEED=METRICS_JSON:PER_CASE_JSON",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for value in args.run:
        seed_text, paths = value.split("=", 1)
        metrics_path, per_case_path = paths.split(":", 1)
        metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
        per_case = json.loads(Path(per_case_path).read_text(encoding="utf-8"))
        rows.append(
            {
                "seed": int(seed_text),
                "metrics_path": metrics_path,
                "per_case_path": per_case_path,
                "metrics": {
                    key: float(metrics["metrics"][key]) for key in METRICS
                },
                "case_count": len(per_case),
                "_per_case": per_case,
            }
        )
    rows.sort(key=lambda row: row["seed"])
    summaries = {
        key: {
            "mean": float(np.mean([row["metrics"][key] for row in rows])),
            "std": float(np.std([row["metrics"][key] for row in rows], ddof=0)),
            "min": float(np.min([row["metrics"][key] for row in rows])),
            "max": float(np.max([row["metrics"][key] for row in rows])),
        }
        for key in METRICS
    }
    reference = rows[0]
    comparisons = {}
    for row in rows[1:]:
        comparisons[f"seed_{row['seed']}_vs_seed_{reference['seed']}"] = {
            key: paired_delta(
                row["_per_case"],
                reference["_per_case"],
                key,
                iterations=2000,
            )
            for key in (
                "fpeval_edit_cost",
                "control_room_pixel_preservation",
                "room_adjacency_f1",
                "conditional_watertight",
            )
            if key in row["_per_case"][0]
            and key in reference["_per_case"][0]
        }
    report = {
        "status": "complete",
        "seeds": [
            {
                key: value
                for key, value in row.items()
                if key != "_per_case"
            }
            for row in rows
        ],
        "summary": summaries,
        "comparisons": comparisons,
        "checks": {
            "all_validity_at_least_0_99": all(
                row["metrics"]["fpeval_validity"] >= 0.99 for row in rows
            ),
            "all_conditional_watertight_at_least_0_95": all(
                row["metrics"]["conditional_watertight"] >= 0.95
                for row in rows
            ),
            "all_positive_edit_cost_reduction": all(
                row["metrics"]["edit_cost_reduction"] > 0.0 for row in rows
            ),
            "all_room_preservation_at_least_0_98": all(
                row["metrics"]["control_room_pixel_preservation"] >= 0.98
                for row in rows
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
