#!/usr/bin/env python3
"""Report whether a global FiLM bottleneck makes the flow causally useful."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_chip_global_rescue_report import training_summary
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
        / "outputs/chip_global_rescue/screen_5k/graph_global_film_seed19_5k",
    )
    parser.add_argument(
        "--baseline-causality",
        type=Path,
        default=ROOT / "outputs/reports/chip_global_causality_probe_v1.json",
    )
    parser.add_argument(
        "--candidate-causality",
        type=Path,
        default=ROOT / "outputs/reports/chip_global_causality_film_v1.json",
    )
    parser.add_argument(
        "--candidate-representation",
        type=Path,
        default=ROOT / "outputs/reports/chip_global_representation_film_v1.json",
    )
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_global_film_screen_v1.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
    causality_baseline = read(args.baseline_causality)
    causality_candidate = read(args.candidate_causality)
    representation = read(args.candidate_representation)["metrics"]
    refined_relative = {
        metric: endpoints["refined"]["paired"][metric]["relative_delta"]
        for metric in ("legalized_hard_cost", "legalized_v_rel")
    }
    checks = {
        "shuffled_global_changes_total_5pct": (
            causality_candidate["effects_vs_original"]["shuffled"][
                "total_relative_delta"
            ]
            >= 0.05
        ),
        "zero_global_changes_total_5pct": (
            causality_candidate["effects_vs_original"]["zero"][
                "total_relative_delta"
            ]
            >= 0.05
        ),
        "oracle_global_improves_total_5pct": (
            causality_candidate["effects_vs_original"]["target_oracle"][
                "total_relative_delta"
            ]
            <= -0.05
        ),
        "candidate_rank_above_8": (
            representation["target_effective_rank"] > 8.0
        ),
        "refined_cost_improves_5pct": (
            refined_relative["legalized_hard_cost"] <= -0.05
        ),
        "refined_v_rel_improves_5pct": (
            refined_relative["legalized_v_rel"] <= -0.05
        ),
    }
    report = {
        "status": "pass" if all(checks.values()) else "reject_global_film",
        "scope": "matched seed-19 5k global statistics plus global FiLM rescue",
        "baseline": training_summary(args.baseline_dir),
        "candidate": training_summary(args.candidate_dir),
        "endpoints": endpoints,
        "causality": {
            "baseline": causality_baseline["effects_vs_original"],
            "candidate": causality_candidate["effects_vs_original"],
        },
        "representation": representation,
        "checks": checks,
        "decision": (
            "promote global FiLM to 20k"
            if all(checks.values())
            else "stop global-path capacity growth and focus on decoder/refiner objectives"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], **checks}, indent=2))


if __name__ == "__main__":
    main()
