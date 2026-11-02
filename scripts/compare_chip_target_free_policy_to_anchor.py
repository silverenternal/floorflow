#!/usr/bin/env python3
"""Compare a frozen target-free candidate policy with a matched anchor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from floorflow.evaluation.paired_bootstrap import paired_delta

METRICS = (
    "legalized_hard_cost",
    "legalized_v_rel",
    "legalized_area_gap",
    "legalized_b2b_hpwl_gap",
    "legalized_p2b_hpwl_gap",
    "legalized_grouping",
    "legalized_boundary",
    "legalized_rho_mae",
    "legalizer_rung",
    "legalizer_solves",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort-dir", type=Path, required=True)
    parser.add_argument("--candidate-rows", type=Path, required=True)
    parser.add_argument("--anchor-cases", type=Path, required=True)
    parser.add_argument("--anchor-key", required=True)
    parser.add_argument(
        "--policy",
        choices=("min_b2b", "min_b2b_safety_guard"),
        default="min_b2b",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=21100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = {
        int(case["case_index"]): case["rows"]
        for case in json.loads(args.candidate_rows.read_text(encoding="utf-8"))[
            "rows"
        ]
    }
    anchor = {
        int(row["index"]): row
        for row in json.loads(args.anchor_cases.read_text(encoding="utf-8"))[
            args.anchor_key
        ]
    }
    choices = {}
    for path in sorted(args.cohort_dir.glob("*.npz")):
        payload = np.load(path)
        index = int(payload["case_index"])
        mask = torch.as_tensor(payload["candidate_mask"]).bool()[0]
        features = torch.as_tensor(payload["candidate_features"]).float()[0][mask]
        if args.policy == "min_b2b":
            choices[index] = int(torch.argmin(features[:, 3]))
            continue
        reference = features[0]
        safe = (
            (features[:, 7] <= reference[7])
            & (features[:, 8] <= reference[8])
            & (features[:, 9] <= 1.1 * reference[9])
            & (features[:, 10] <= 1.1 * reference[10])
        )
        score = features[:, 3].clone()
        score[~safe] = float("inf")
        score[0] = features[0, 3]
        choices[index] = int(torch.argmin(score))
    indices = sorted(set(candidates) & set(anchor) & set(choices))
    selected = [
        {
            **candidates[index][f"candidate_{choices[index]}"],
            "index": index,
        }
        for index in indices
    ]
    reference = [{**anchor[index], "index": index} for index in indices]
    paired = {
        metric: paired_delta(
            selected,
            reference,
            metric,
            iterations=args.bootstrap,
            seed=args.seed + offset,
            lower_is_better=True,
        )
        for offset, metric in enumerate(METRICS)
    }
    wins = {"joint": 0, "strict_structural": 0, "constraint_safe": 0}
    for current, baseline in zip(selected, reference, strict=True):
        joint = (
            current["legalized_hard_cost"]
            <= baseline["legalized_hard_cost"]
            and current["legalized_v_rel"] <= baseline["legalized_v_rel"]
        )
        strict = (
            joint
            and current["legalized_grouping"]
            <= baseline["legalized_grouping"]
            and current["legalized_boundary"]
            <= baseline["legalized_boundary"]
            and current["legalized_p2b_hpwl_gap"]
            <= baseline["legalized_p2b_hpwl_gap"]
        )
        safe = (
            strict
            and current["legalized_area_gap"]
            <= baseline["legalized_area_gap"]
            and current["legalized_b2b_hpwl_gap"]
            <= baseline["legalized_b2b_hpwl_gap"]
        )
        wins["joint"] += int(joint)
        wins["strict_structural"] += int(strict)
        wins["constraint_safe"] += int(safe)
    report = {
        "version": "chip-target-free-policy-anchor-v1",
        "status": "complete",
        "policy": args.policy,
        "cases": len(indices),
        "candidate_rows": str(args.candidate_rows),
        "anchor_cases": str(args.anchor_cases),
        "wins": wins,
        "means": {
            "selected": {
                metric: float(np.mean([row[metric] for row in selected]))
                for metric in METRICS
            },
            "anchor": {
                metric: float(np.mean([row[metric] for row in reference]))
                for metric in METRICS
            },
        },
        "paired": paired,
        "claim_boundary": (
            "Target-free candidate selection versus a matched anchor; "
            "all candidate and legalizer calls must be charged to the portfolio."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"cases": len(indices), "wins": wins}, indent=2))


if __name__ == "__main__":
    main()
