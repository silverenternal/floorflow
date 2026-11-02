#!/usr/bin/env python3
"""Report the global-latent information-preservation rescue screen."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_chip_graph_mechanism_report import (
    endpoint_rows,
    endpoint_summary,
    paired_metrics,
    read,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=ROOT
        / "outputs/chip_graph_mechanism/screen_5k/graph_full_seed19_5k",
    )
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=ROOT
        / "outputs/chip_global_rescue/screen_5k/graph_global_stats_v2_seed19_5k",
    )
    parser.add_argument(
        "--baseline-representation",
        type=Path,
        default=ROOT / "outputs/reports/chip_global_representation_probe_v1.json",
    )
    parser.add_argument(
        "--candidate-representation",
        type=Path,
        default=ROOT
        / "outputs/reports/chip_global_representation_stats_v1.json",
    )
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_global_pool_stats_screen_v1.json",
    )
    return parser.parse_args()


def training_summary(directory: Path) -> dict:
    report = read(directory.parent / f"{directory.name}.json")
    return {
        "parameters": int(report["model_parameters"]),
        "dev_total": float(report["final_dev"]["total"]),
        "final_dev": report["final_dev"],
    }


def main() -> None:
    args = parse_args()
    baseline_training = training_summary(args.baseline_dir)
    candidate_training = training_summary(args.candidate_dir)
    endpoints = {}
    for kind in ("raw", "refined"):
        endpoints[kind] = {
            "baseline": endpoint_summary(args.baseline_dir, kind),
            "candidate": endpoint_summary(args.candidate_dir, kind),
            "paired": paired_metrics(
                endpoint_rows(args.baseline_dir, kind),
                endpoint_rows(args.candidate_dir, kind),
                iterations=args.bootstrap,
            ),
        }
    baseline_rep = read(args.baseline_representation)["metrics"]
    candidate_rep = read(args.candidate_representation)["metrics"]
    representation = {
        "baseline_target_std": baseline_rep["target_dim_std_mean"],
        "candidate_target_std": candidate_rep["target_dim_std_mean"],
        "target_std_ratio": (
            candidate_rep["target_dim_std_mean"]
            / baseline_rep["target_dim_std_mean"]
        ),
        "baseline_target_effective_rank": baseline_rep[
            "target_effective_rank"
        ],
        "candidate_target_effective_rank": candidate_rep[
            "target_effective_rank"
        ],
    }
    refined_relative = {
        metric: endpoints["refined"]["paired"][metric]["relative_delta"]
        for metric in ("legalized_hard_cost", "legalized_v_rel")
    }
    checks = {
        "target_std_recovers_2x": representation["target_std_ratio"] >= 2.0,
        "target_rank_does_not_drop": (
            representation["candidate_target_effective_rank"]
            >= representation["baseline_target_effective_rank"]
        ),
        "refined_cost_improves_5pct": (
            refined_relative["legalized_hard_cost"] <= -0.05
        ),
        "refined_v_rel_improves_5pct": (
            refined_relative["legalized_v_rel"] <= -0.05
        ),
    }
    report = {
        "status": "pass" if all(checks.values()) else "reject_global_stats_skip",
        "scope": "matched seed-19 5k global-latent information-preservation rescue",
        "baseline": baseline_training,
        "candidate": candidate_training,
        "endpoints": endpoints,
        "representation": representation,
        "checks": checks,
        "decision": (
            "promote global-statistics skip to 20k"
            if all(checks.values())
            else "retain the established graph global-slot model and investigate decoder-level bottlenecks"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], **checks}, indent=2))


if __name__ == "__main__":
    main()
