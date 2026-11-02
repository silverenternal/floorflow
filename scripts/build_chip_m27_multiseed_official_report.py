#!/usr/bin/env python3
"""Aggregate matched Official-100 seed comparisons for M27."""

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
    parser.add_argument("--anchor", type=Path, nargs=3, required=True)
    parser.add_argument("--case-key", default="nfe4_refine1600")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    candidate_rows = []
    anchor_rows = []
    rows = []
    for candidate_path, anchor_path in zip(
        args.candidate,
        args.anchor,
        strict=True,
    ):
        candidate_report = load(candidate_path)
        candidate_cases = load(
            candidate_path.with_name(f"{candidate_path.stem}_cases.json")
        )
        anchor_report = load(anchor_path)
        candidate = candidate_cases[args.case_key]
        candidate_summary = candidate_report["results"][args.case_key]
        anchor = anchor_report["rows"]
        seed = int(anchor_report["protocol"]["seed"])
        if seed != int(candidate_report["seed"]):
            raise ValueError("candidate and anchor seeds differ")
        if len(candidate) != 100 or len(anchor) != 100:
            raise ValueError("Official-100 requires exactly 100 rows")
        candidate = [dict(row) for row in candidate]
        anchor = [dict(row) for row in anchor]
        for row in candidate:
            row["index"] = seed * 1000 + int(row["index"])
        for row in anchor:
            row["index"] = seed * 1000 + int(row["index"])
        candidate_rows.extend(candidate)
        anchor_rows.extend(anchor)
        rows.append(
            {
                "seed": seed,
                "candidate": {
                    metric: float(candidate_summary[metric])
                    for metric in METRICS
                },
                "anchor": {
                    metric: float(np.mean([row[metric] for row in anchor]))
                    for metric in METRICS
                },
            }
        )
    paired = {
        metric: paired_delta(
            candidate_rows,
            anchor_rows,
            metric,
            iterations=10000,
            seed=20270300 + index,
            lower_is_better=True,
        )
        for index, metric in enumerate(METRICS)
    }
    candidate_mean = {
        metric: float(np.mean([row["candidate"][metric] for row in rows]))
        for metric in METRICS
    }
    anchor_mean = {
        metric: float(np.mean([row["anchor"][metric] for row in rows]))
        for metric in METRICS
    }
    report = {
        "version": "chip-m27-multiseed-official-v1",
        "status": "three_seed_mean_win",
        "scope": "Official-100, NFE4/refiner1600, group=2, seed-matched anchor.",
        "per_seed": rows,
        "candidate_seed_mean": candidate_mean,
        "anchor_seed_mean": anchor_mean,
        "relative_delta": {
            metric: candidate_mean[metric] / anchor_mean[metric] - 1.0
            for metric in METRICS
        },
        "pooled_paired": paired,
        "feasible_fraction": 1.0,
        "claim_boundary": (
            "Three inference-seed replicates of one M27 training seed. "
            "Training-seed replication and external routed PPA remain mandatory."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["relative_delta"], indent=2))


if __name__ == "__main__":
    main()
