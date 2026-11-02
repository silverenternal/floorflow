#!/usr/bin/env python3
"""Compare two opening decoder reports with per-case paired statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from floorflow.evaluation.paired_bootstrap import paired_delta

METRICS = (
    "pixel_f1",
    "vector_precision",
    "vector_recall",
    "vector_f1",
    "predicted_openings",
    "matched_openings",
)


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
        raise ValueError("opening reports have no shared cases")
    paired = {}
    for metric in METRICS:
        current = [{"value": row[metric]} for row, _base in rows]
        reference = [{"value": base[metric]} for _row, base in rows]
        paired[metric] = paired_delta(
            current,
            reference,
            "value",
            iterations=args.bootstrap,
            seed=17,
        )
    gate = {
        "vector_f1_ci_lower_positive": paired["vector_f1"]["ci_low"] > 0.0,
        "pixel_f1_noninferiority": paired["pixel_f1"]["mean_delta"] >= -0.002,
    }
    gate["promoted"] = all(gate.values())
    report = {
        "schema_version": "opening-checkpoint-paired-comparison-v1",
        "candidate": str(args.candidate),
        "baseline": str(args.baseline),
        "cases": len(rows),
        "paired": paired,
        "gate": gate,
        "decision": (
            "promote_boundary_context"
            if gate["promoted"]
            else "retain_as_diagnostic"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"paired": paired, "gate": gate}, indent=2))


if __name__ == "__main__":
    main()
