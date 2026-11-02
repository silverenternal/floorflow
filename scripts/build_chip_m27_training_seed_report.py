#!/usr/bin/env python3
"""Aggregate fixed-inference Official-100 results across M27 training seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from floorflow.evaluation.paired_bootstrap import paired_delta

ROOT = Path(__file__).resolve().parents[1]
METRICS = (
    "legalized_hard_cost",
    "legalized_v_rel",
    "legalized_area_gap",
    "legalized_b2b_hpwl_gap",
    "legalized_p2b_hpwl_gap",
    "legalized_grouping",
    "legalized_boundary",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, nargs=3, required=True)
    parser.add_argument(
        "--training-seed",
        type=int,
        nargs=3,
        required=True,
    )
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--case-key", default="nfe4_refine1600")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def prefixed(rows: list[dict], seed: int) -> list[dict]:
    copied = [dict(row) for row in rows]
    for row in copied:
        row["index"] = seed * 1000 + int(row["index"])
    return copied


def main() -> None:
    args = parse_args()
    args.candidate = [path.resolve() for path in args.candidate]
    args.anchor = args.anchor.resolve()
    args.output = args.output.resolve()
    anchor_report = load(args.anchor)
    anchor_rows = prefixed(anchor_report["rows"], 0)
    candidate_rows = []
    per_training_seed = []
    anchor_mean = {
        metric: float(np.mean([row[metric] for row in anchor_rows]))
        for metric in METRICS
    }
    for path, seed in zip(args.candidate, args.training_seed, strict=True):
        report = load(path)
        cases = load(path.with_name(f"{path.stem}_cases.json"))
        rows = prefixed(cases[args.case_key], seed)
        candidate_rows.extend(rows)
        per_training_seed.append(
            {
                "training_seed": seed,
                "report": str(path.relative_to(ROOT)),
                "metrics": {
                    metric: float(report["results"][args.case_key][metric])
                    for metric in METRICS
                },
                "cost_relative": (
                    float(report["results"][args.case_key]["legalized_hard_cost"])
                    / anchor_mean["legalized_hard_cost"]
                    - 1.0
                ),
                "v_rel_relative": (
                    float(report["results"][args.case_key]["legalized_v_rel"])
                    / anchor_mean["legalized_v_rel"]
                    - 1.0
                ),
            }
        )
    repeated_anchor = [
        row
        for seed in args.training_seed
        for row in prefixed(anchor_report["rows"], seed)
    ]
    paired = {
        metric: paired_delta(
            candidate_rows,
            repeated_anchor,
            metric,
            iterations=10000,
            seed=20280400 + index,
            lower_is_better=True,
        )
        for index, metric in enumerate(METRICS)
    }
    candidate_mean = {
        metric: float(np.mean([row[metric] for row in candidate_rows]))
        for metric in METRICS
    }
    report = {
        "version": "chip-m27-training-seed-v1",
        "status": "three_training_seed_fixed_inference",
        "scope": (
            "Official-100, NFE4/refiner1600, group=2, inference seed 19, "
            "training seeds 19/20/21."
        ),
        "anchor": str(args.anchor.relative_to(ROOT)),
        "training_seeds": list(args.training_seed),
        "per_training_seed": per_training_seed,
        "candidate_training_seed_mean": candidate_mean,
        "anchor_mean": anchor_mean,
        "relative_delta": {
            metric: candidate_mean[metric] / anchor_mean[metric] - 1.0
            for metric in METRICS
        },
        "pooled_paired": paired,
        "feasible_fraction": 1.0,
        "claim_boundary": "Training-seed evidence only; full-data and external PPA remain pending.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["relative_delta"], indent=2))


if __name__ == "__main__":
    main()
