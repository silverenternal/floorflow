#!/usr/bin/env python3
"""Aggregate three-seed graph-token flow evidence against graph-slot controls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from floorflow.evaluation.paired_bootstrap import paired_delta
from scripts.build_chip_trinity_token_screen_report import (
    endpoint_metrics,
    endpoint_report,
    endpoint_rows,
    read,
)

SEEDS = (19, 20, 21)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate-root",
        type=Path,
        default=ROOT / "outputs/chip_trinity/rescue_20k",
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=ROOT / "outputs/chip_final/screen_20k",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_trinity_token_three_seed_v1.json",
    )
    parser.add_argument("--bootstrap", type=int, default=4000)
    return parser.parse_args()


def arm_paths(root: Path, seed: int) -> tuple[Path, Path]:
    if root.name == "rescue_20k":
        candidate = root / f"trinity_token_seed{seed}_20k"
        training = root / f"trinity_token_seed{seed}_20k.json"
    else:
        candidate = root / f"graph_velocity_seed{seed}_20k_final"
        training = root / f"graph_velocity_seed{seed}_20k_final.json"
    return candidate, training


def mean_metric(rows: list[dict], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def strata(rows: list[dict]) -> dict:
    output = {}
    for name, selected in (
        ("21-60", [row for row in rows if int(row["blocks"]) <= 60]),
        ("61-120", [row for row in rows if int(row["blocks"]) > 60]),
    ):
        output[name] = {
            "cases": len(selected),
            "legalized_hard_cost": mean_metric(selected, "legalized_hard_cost"),
            "legalized_v_rel": mean_metric(selected, "legalized_v_rel"),
        }
    return output


def compare_rows(
    candidate_rows: list[dict],
    baseline_rows: list[dict],
    *,
    bootstrap: int,
    seed: int,
) -> dict:
    return {
        metric: paired_delta(
            candidate_rows,
            baseline_rows,
            metric,
            iterations=bootstrap,
            seed=20261110 + seed * 10 + index,
            lower_is_better=True,
        )
        for index, metric in enumerate(
            ("legalized_hard_cost", "legalized_v_rel")
        )
    }


def main() -> None:
    args = parse_args()
    seeds = {}
    for seed in SEEDS:
        candidate_dir, candidate_training_path = arm_paths(
            args.candidate_root,
            seed,
        )
        baseline_dir, baseline_training_path = arm_paths(
            args.baseline_root,
            seed,
        )
        candidate_training = read(candidate_training_path)
        baseline_training = read(baseline_training_path)
        seed_rows = {
            "candidate": {
                "parameters": int(candidate_training["model_parameters"]),
                "dev": candidate_training["final_dev"],
            },
            "baseline": {
                "parameters": int(baseline_training["model_parameters"]),
                "dev": baseline_training["final_dev"],
            },
            "endpoints": {},
        }
        for kind in ("raw", "refined"):
            candidate_summary = endpoint_metrics(
                endpoint_report(candidate_dir, kind)
            )
            baseline_summary = endpoint_metrics(
                endpoint_report(baseline_dir, kind)
            )
            candidate_rows = endpoint_rows(candidate_dir, kind)
            baseline_rows = endpoint_rows(baseline_dir, kind)
            seed_rows["endpoints"][kind] = {
                "candidate": candidate_summary,
                "baseline": baseline_summary,
                "relative_delta": {
                    metric: candidate_summary[metric]
                    / baseline_summary[metric]
                    - 1.0
                    for metric in ("legalized_hard_cost", "legalized_v_rel")
                },
                "paired": compare_rows(
                    candidate_rows,
                    baseline_rows,
                    bootstrap=args.bootstrap,
                    seed=seed,
                ),
                "strata": {
                    "candidate": strata(candidate_rows),
                    "baseline": strata(baseline_rows),
                },
            }
        seeds[str(seed)] = seed_rows

    summary = {}
    for kind in ("raw", "refined"):
        per_seed = [
            seeds[str(seed)]["endpoints"][kind]["relative_delta"]
            for seed in SEEDS
        ]
        summary[kind] = {
            "seed_mean_relative_delta": {
                metric: sum(row[metric] for row in per_seed) / len(per_seed)
                for metric in ("legalized_hard_cost", "legalized_v_rel")
            },
            "absolute": {
                role: {
                    metric: sum(
                        seeds[str(seed)]["endpoints"][kind][role][metric]
                        for seed in SEEDS
                    )
                    / len(SEEDS)
                    for metric in (
                        "legalized_hard_cost",
                        "legalized_v_rel",
                    )
                }
                for role in ("candidate", "baseline")
            },
        }
    checks = {
        "candidate_wins_both_refined_metrics_all_seeds": all(
            seeds[str(seed)]["endpoints"]["refined"]["relative_delta"][metric]
            < 0.0
            for seed in SEEDS
            for metric in ("legalized_hard_cost", "legalized_v_rel")
        ),
        "candidate_wins_both_raw_metrics_all_seeds": all(
            seeds[str(seed)]["endpoints"]["raw"]["relative_delta"][metric] < 0.0
            for seed in SEEDS
            for metric in ("legalized_hard_cost", "legalized_v_rel")
        ),
        "seed_mean_refined_improves_both": all(
            value < 0.0
            for value in summary["refined"]["seed_mean_relative_delta"].values()
        ),
    }
    report = {
        "status": "pass" if all(checks.values()) else "fail_seed_stability",
        "scope": "three-seed 20k graph-token flow versus frozen graph-slot controls",
        "protocol": {
            "seeds": list(SEEDS),
            "raw": "NFE8, scale_pack legalization, score_full_fast",
            "refined": "NFE4, 400-step closed-form refiner, score_full_fast",
            "best_of_n_used": False,
        },
        "seeds": seeds,
        "summary": summary,
        "checks": checks,
        "claim_boundary": (
            "Use the measured seed-mean effect and report the paired long-tail "
            "intervals; do not select cases or compare against a different protocol."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
