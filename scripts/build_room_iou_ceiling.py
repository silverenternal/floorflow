#!/usr/bin/env python3
"""Assemble the R1 ambiguity and room-IoU ceiling report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np

from floorflow.inference.sampling_policy import best_of_n_curve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ambiguity", type=Path, required=True)
    parser.add_argument("--matching", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ambiguity = json.loads(args.ambiguity.read_text(encoding="utf-8"))
    matching = json.loads(args.matching.read_text(encoding="utf-8"))
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    if matching.get("best_of_n_hungarian_iou") is None:
        raise ValueError("room matching input must be generated with --diagnostic-ceiling")
    candidate_rows = [row.get("candidates") for row in matching.get("rows", [])]
    n_curve = {}
    if candidate_rows and all(values is not None for values in candidate_rows):
        n_curve = {
            str(budget): value
            for budget, value in best_of_n_curve(np.asarray(candidate_rows, dtype=np.float64)).items()
        }
    report = {
        "claim_role": "ambiguity_and_expression_ceiling_diagnostic",
        "condition_entropy": ambiguity["mean_condition_entropy"],
        "neighborhood_area_mae": ambiguity["mean_neighborhood_area_mae"],
        "single_layout_room_macro_iou": metrics["room_macro_iou_mean"],
        "top1_hungarian_iou": matching["top1_hungarian_iou"],
        "best_of_n_hungarian_iou": matching["best_of_n_hungarian_iou"],
        "best_of_n_samples": matching["samples"],
        "best_of_n_usage": "diagnostic_only_not_primary_claim",
        "expected_hungarian_iou": matching.get("expected_hungarian_iou"),
        "sampling_policy": matching.get("sampling_policy"),
        "diagnostic_n_curve": n_curve,
        "interpretation": {
            "one_to_many_confirmed": ambiguity["mean_condition_entropy"] > 1.0,
            "sample_max_gap": matching["best_of_n_hungarian_iou"] - matching["top1_hungarian_iou"],
            "primary_metric": "conditional distribution distance and expected structural validity",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
