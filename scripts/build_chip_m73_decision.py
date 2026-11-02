#!/usr/bin/env python3
"""Compare the M73 cross-checkpoint portfolio with matched Trinity d192."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from floorflow.evaluation.paired_bootstrap import paired_delta

CROSS = (
    "outputs/reports/chip_m73_cross_checkpoint_portfolio_official100_v1.json"
)
TRINITY = "outputs/reports/chip_trinity_d192_nfe4_ref3200_official100_v1.json"
METRICS = (
    "cost",
    "v_rel",
    "area_gap",
    "b2b_hpwl_gap",
    "p2b_hpwl_gap",
    "grouping",
    "boundary",
    "rho_mae",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_m73_official100_decision_v1.json",
    )
    parser.add_argument("--cross", type=Path, default=ROOT / CROSS)
    parser.add_argument("--trinity", type=Path, default=ROOT / TRINITY)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--bootstrap", type=int, default=10000)
    return parser.parse_args()


def load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    cross = load(args.cross)
    candidate = cross["selected_rows"]
    reference = [
        {
            "index": int(row["index"]),
            "cost": float(row["legalized_hard_cost"]),
            "v_rel": float(row["legalized_v_rel"]),
            "area_gap": float(row["legalized_area_gap"]),
            "b2b_hpwl_gap": float(row["legalized_b2b_hpwl_gap"]),
            "p2b_hpwl_gap": float(row["legalized_p2b_hpwl_gap"]),
            "grouping": float(row["legalized_grouping"]),
            "boundary": float(row["legalized_boundary"]),
            "rho_mae": float(row["legalized_rho_mae"]),
            "feasible": bool(row["feasible"]),
        }
        for row in load(args.trinity)["rows"]
    ]
    if len(candidate) != 100 or len(reference) != 100:
        raise ValueError("official M73 decision requires 100 paired cases")
    paired = {
        metric: paired_delta(
            candidate,
            reference,
            metric,
            iterations=args.bootstrap,
            seed=7300 + offset,
            lower_is_better=True,
        )
        for offset, metric in enumerate(METRICS)
    }
    means = {
        metric: {
            "candidate": sum(float(row[metric]) for row in candidate) / 100,
            "trinity": sum(float(row[metric]) for row in reference) / 100,
        }
        for metric in METRICS
    }
    for metric in METRICS:
        means[metric]["delta"] = (
            means[metric]["candidate"] - means[metric]["trinity"]
        )
    checks = {
        "feasibility_one": all(
            bool(row["feasible"]) for row in candidate
        ),
        "cost_mean_better": means["cost"]["delta"] < 0.0,
        "v_rel_mean_better": means["v_rel"]["delta"] < 0.0,
        "cost_ci_excludes_zero": paired["cost"]["ci_high"] < 0.0,
        "v_rel_ci_excludes_zero": paired["v_rel"]["ci_high"] < 0.0,
    }
    report = {
        "version": "chip-m73-official100-decision-v1",
        "status": "pass_mechanism_and_mean_win"
        if checks["cost_mean_better"]
        and checks["v_rel_mean_better"]
        and checks["feasibility_one"]
        else "fail",
        "cross_checkpoint_report": str(args.cross),
        "trinity_report": str(args.trinity),
        "seed": int(args.seed),
        "means": means,
        "paired": paired,
        "checks": checks,
        "budget": {
            "checkpoints": 3,
            "trajectories": 6,
            "nfe_total": 10,
            "refiner_steps_total": 9600,
            "legalizer_calls": 9,
            "note": (
                "M48 contributes two NFE1/refiner1600 trajectories; M61B and M65 "
                "each contribute two NFE2/refiner1600 trajectories. Combined NFE "
                "is 10 versus 4 for the one-trajectory Trinity anchor."
            ),
        },
        "selection_counts": cross["selection_counts"],
        "claim_boundary": (
            "Mean win against matched neural-budget Trinity, but the cost interval "
            "must be checked and the six-trajectory/nine-legalizer budget disclosed."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], **checks}, indent=2))


if __name__ == "__main__":
    main()
