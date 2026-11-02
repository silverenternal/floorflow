#!/usr/bin/env python3
"""Freeze the expanded-range global-slot adaptation screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from floorflow.evaluation.paired_bootstrap import paired_delta

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument(
        "--baseline-cases",
        type=Path,
        default=ROOT
        / "outputs/chip_slot/global_v1/global_slot_overlap10_20k/"
        "official_test100_nfe8_legalized_cases.json",
    )
    parser.add_argument(
        "--candidate-cases",
        type=Path,
        default=ROOT
        / "outputs/chip_range_adapt/seed19/global_range_resume5k/"
        "official_test100_nfe8_legalized_cases.json",
    )
    parser.add_argument(
        "--baseline-refiner",
        type=Path,
        default=ROOT / "outputs/reports/chip_inference_refiner_comparison_v1_cases.json",
    )
    parser.add_argument(
        "--candidate-refiner",
        type=Path,
        default=ROOT
        / "outputs/chip_range_adapt/seed19/global_range_resume5k/"
        "refiner_nfe4_cases.json",
    )
    parser.add_argument(
        "--scope",
        default="seed-19 5k continuation on 200k train cases covering 21-120 blocks",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_range_adaptation_screen_v1.json",
    )
    return parser.parse_args()


def read(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def paired(
    control: list[dict],
    candidate: list[dict],
    metric: str,
    *,
    iterations: int,
) -> dict:
    result = paired_delta(
        candidate,
        control,
        metric,
        iterations=iterations,
        seed=20261027,
        lower_is_better=True,
    )
    return {
        "cases": result["cases"],
        "control_mean": result["baseline_mean"],
        "candidate_mean": result["baseline_mean"] + result["mean_delta"],
        "paired_delta": result["mean_delta"],
        "relative_delta": result["relative_delta"],
        "ci95": [result["ci_low"], result["ci_high"]],
    }


def main() -> None:
    args = parse_args()
    original_cases = read(args.baseline_cases)
    adapted_cases = read(args.candidate_cases)
    original_refiner = read(args.baseline_refiner)
    adapted_refiner = read(args.candidate_refiner)
    if "nfe4_refine400" in original_refiner:
        original_refiner = original_refiner["nfe4_refine400"]
    if "nfe4_refine400" in adapted_refiner:
        adapted_refiner = adapted_refiner["nfe4_refine400"]
    metrics = {}
    for label, left, right in (
        ("raw_legalized", original_cases, adapted_cases),
        ("refined", original_refiner, adapted_refiner),
    ):
        metrics[label] = {
            "legalized_hard_cost": paired(
                left,
                right,
                "legalized_hard_cost",
                iterations=args.bootstrap,
            ),
            "legalized_v_rel": paired(
                left,
                right,
                "legalized_v_rel",
                iterations=args.bootstrap,
            ),
        }
    checks = {
        f"{stage}_{metric}_improves": (
            values[metric]["paired_delta"] < 0.0
        )
        for stage, values in metrics.items()
        for metric in ("legalized_hard_cost", "legalized_v_rel")
    }
    by_block = {}
    for name, indices in (
        ("21-60", [i for i, row in enumerate(original_cases) if row["blocks"] <= 60]),
        ("61-120", [i for i, row in enumerate(original_cases) if row["blocks"] > 60]),
    ):
        by_block[name] = {
            metric: {
                "control": float(np.mean([original_cases[i][metric] for i in indices])),
                "candidate": float(np.mean([adapted_cases[i][metric] for i in indices])),
            }
            for metric in ("legalized_hard_cost", "legalized_v_rel")
        }
        for values in by_block[name].values():
            values["relative_delta"] = (
                values["candidate"] - values["control"]
            ) / values["control"]
        checks[f"long_stratum_{name}_cost_improves"] = (
            by_block[name]["legalized_hard_cost"]["relative_delta"] < 0.0
        )
        checks[f"long_stratum_{name}_v_rel_improves"] = (
            by_block[name]["legalized_v_rel"]["relative_delta"] < 0.0
        )
    accepted = all(checks.values())
    report = {
        "status": "pass" if accepted else "reject_range_adaptation",
        "scope": args.scope,
        "checks": checks,
        "paired": metrics,
        "by_block": by_block,
        "protocol": {
            "raw": "NFE8, scale_pack, score_full_fast",
            "refined": "NFE4, 400 closed-form refiner steps, scale_pack",
            "best_of_n_used": False,
        },
        "dense_control": {
            "status": "not_run_after_global_gate_failure",
            "reason": "The global arm failed both raw and refined V_rel checks; stopping avoids an uninformative GPU run.",
        },
        "decision": (
            "run U3 replication"
            if accepted
            else "reject expanded-range continuation; keep the original three-seed 50k result"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "checks": checks,
                "decision": report["decision"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
