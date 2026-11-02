#!/usr/bin/env python3
"""Evaluate a distilled candidate-consensus fuser on frozen legalized cases."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/Trinity/src"))

from floorflow.chip_flow.candidate_consensus_evaluation import (
    evaluate_candidate_consensus_case,
    load_candidate_consensus_case,
)
from floorflow.chip_flow.candidate_consensus_training import (
    load_candidate_consensus_checkpoint,
)
from floorflow.chip_flow.legalizer_policy import LegalizerPolicy
from scripts.evaluate_chip_official import score_case


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--projector-iterations", type=int, default=2)
    parser.add_argument("--legalizer-time-limit-ms", type=int, default=2000)
    parser.add_argument("--legalizer-policy-json", type=Path)
    parser.add_argument("--promotion-min-cases", type=int, default=500)
    return parser.parse_args()


def _aggregate(values: list[float | None]) -> dict:
    finite = [float(value) for value in values if value is not None]
    return {
        "mean": float(np.mean(finite)) if finite else None,
        "valid_cases": len(finite),
    }


def main() -> None:
    args = parse_args()
    policy = (
        LegalizerPolicy(
            **json.loads(args.legalizer_policy_json.read_text(encoding="utf-8"))
        )
        if args.legalizer_policy_json is not None
        else None
    )
    if policy is not None:
        policy.validate()
    model = load_candidate_consensus_checkpoint(args.checkpoint)
    files = sorted(args.cohort_dir.glob("*.npz"))
    if not files:
        raise ValueError("candidate consensus cohort contains no shards")
    rows = [
        evaluate_candidate_consensus_case(
            model,
            load_candidate_consensus_case(path),
            score_case,
            projector_iterations=args.projector_iterations,
            legalizer_time_limit_ms=args.legalizer_time_limit_ms,
            legalizer_options=policy.kwargs() if policy is not None else None,
        )
        for path in files
    ]
    comparisons = {}
    for baseline in ("reference", "best_single"):
        key = "consensus_vs_" + baseline
        metrics = rows[0][key]
        comparisons[baseline] = {
            metric: _aggregate([row[key][metric] for row in rows])
            for metric in metrics
        }
    report = {
        "version": "chip-candidate-consensus-evaluation-v1",
        "status": (
            "complete"
            if len(rows) >= args.promotion_min_cases
            else "pilot_complete_no_promotion"
        ),
        "scope": "legalized paired evaluation",
        "cases": len(rows),
        "promotion_eligible": len(rows) >= args.promotion_min_cases,
        "promotion_reason": (
            "minimum cohort reached"
            if len(rows) >= args.promotion_min_cases
            else f"{len(rows)} cases is below {args.promotion_min_cases}"
        ),
        "comparisons": comparisons,
        "rows": rows,
        "contract": "same projector and legalizer for every candidate and consensus",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "cases",
                    "promotion_eligible",
                    "comparisons",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
