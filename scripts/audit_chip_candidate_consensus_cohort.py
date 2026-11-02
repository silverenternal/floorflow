#!/usr/bin/env python3
"""Audit candidate diversity and teacher coverage in a consensus cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from floorflow.chip_flow.candidate_consensus_training import (
    load_candidate_consensus_batch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-cases", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = sorted(args.cohort_dir.glob("*.npz"))
    if not files:
        raise ValueError("candidate consensus cohort contains no shards")
    qualities = []
    best_counts = []
    diversity = []
    candidate_counts = []
    invalid_valid_rows = 0
    for path in files:
        batch = load_candidate_consensus_batch(path)
        valid = batch.candidate_mask[0]
        invalid_valid_rows += int(
            (~torch.isfinite(batch.candidate_states[0][valid])).any()
        )
        quality = batch.quality[0].masked_fill(~valid, float("inf"))
        best_counts.append(int(quality.argmin()))
        qualities.append(batch.quality[0][valid].cpu().numpy())
        candidate_counts.append(int(valid.sum()))
        states = batch.candidate_states[0][valid]
        pairwise = torch.cdist(
            states.reshape(states.shape[0], -1),
            states.reshape(states.shape[0], -1),
        )
        count = states.shape[0]
        diversity.append(
            float(pairwise[~torch.eye(count, dtype=torch.bool)].mean())
        )
    candidate_total = max(candidate_counts)
    counts = np.bincount(best_counts, minlength=candidate_total)
    report = {
        "version": "chip-candidate-consensus-cohort-audit-v1",
        "status": "pass" if invalid_valid_rows == 0 else "fail",
        "cohort_dir": str(args.cohort_dir),
        "cases": len(files),
        "candidate_count": candidate_total,
        "minimum_candidate_count": min(candidate_counts),
        "best_single_counts": counts.tolist(),
        "reference_win_rate": float(counts[0] / len(files)),
        "quality_mean": np.mean(np.stack(qualities), axis=0).tolist(),
        "quality_std": np.std(np.stack(qualities), axis=0).tolist(),
        "pairwise_latent_distance_mean": float(np.mean(diversity)),
        "pairwise_latent_distance_p95": float(np.quantile(diversity, 0.95)),
        "promotion_eligible": len(files) >= args.minimum_cases,
        "promotion_reason": (
            "minimum cohort reached"
            if len(files) >= args.minimum_cases
            else f"{len(files)} cases is below {args.minimum_cases}"
        ),
        "checks": {
            "at_least_four_candidates": min(candidate_counts) >= 4,
            "all_cases_have_valid_reference": all(
                load_candidate_consensus_batch(path).candidate_mask[0, 0]
                for path in files
            ),
            "multiple_teacher_winners": int((counts > 0).sum()) >= 2,
            "nontrivial_candidate_diversity": float(np.mean(diversity)) > 0.0,
            "no_invalid_valid_row": invalid_valid_rows == 0,
        },
    }
    report["pipeline_checks_pass"] = all(report["checks"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "cases",
                    "candidate_count",
                    "best_single_counts",
                    "reference_win_rate",
                    "promotion_eligible",
                    "pipeline_checks_pass",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
