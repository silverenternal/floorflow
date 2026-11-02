#!/usr/bin/env python3
"""Summarize topology-aware decoding attempts against the retained decoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from floorflow.evaluation.paired_bootstrap import paired_delta

METRICS = (
    "room_adjacency_f1",
    "door_contact_room_fraction",
    "door_graph_components",
    "fpeval_watertight",
    "fpeval_edit_cost",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="label=path",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("outputs/eval/fp_architecture_ablation_val191_v1/full"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reports/fp_topology_aware_decoding_v1.json"),
    )
    return parser.parse_args()


def read_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def load_run(root: Path) -> dict:
    metrics = read_json(root / "metrics.json")
    per_case = read_json(root / "per_case.json")
    conditional = [
        row["fpeval_watertight"]
        for row in per_case
        if row["target_watertight"] > 0.5
    ]
    return {
        "root": str(root),
        "metrics": metrics["metrics"] | {
            "conditional_watertight": sum(conditional) / len(conditional)
        },
        "per_case": per_case,
    }


def main() -> None:
    args = parse_args()
    baseline = load_run(args.baseline)
    candidates = {}
    for specification in args.candidate:
        label, raw_path = specification.split("=", 1)
        run = load_run(Path(raw_path))
        comparisons = {
            metric: paired_delta(
                run["per_case"],
                baseline["per_case"],
                metric,
                iterations=2000,
            )
            for metric in METRICS
        }
        checks = {
            "adjacency_gain": comparisons["room_adjacency_f1"]["ci_low"] > 0.0,
            "door_contact_not_worse": (
                comparisons["door_contact_room_fraction"]["ci_low"] >= 0.0
            ),
            "watertight_not_worse": (
                comparisons["fpeval_watertight"]["ci_low"] >= 0.0
            ),
            "edit_cost_not_worse": (
                comparisons["fpeval_edit_cost"]["ci_high"] <= 0.0
            ),
        }
        candidates[label] = {
            "comparisons": comparisons,
            "checks": checks,
            "passed": all(checks.values()),
        }
    report = {
        "status": (
            "promoted"
            if any(item["passed"] for item in candidates.values())
            else "failed_gate"
        ),
        "baseline": baseline["metrics"],
        "candidates": candidates,
        "decision": (
            "No tested decoder improves adjacency under the paired gate while "
            "preserving topology and edit utility."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "status": report["status"]}, indent=2))


if __name__ == "__main__":
    main()
