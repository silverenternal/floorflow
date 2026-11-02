#!/usr/bin/env python3
"""Report the two-step sampling-path consistency objective screen."""

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
        / "outputs/chip_global_rescue/screen_5k/graph_path2_seed19_5k",
    )
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_path_consistency_screen_v1.json",
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
    relative = {
        metric: endpoints["refined"]["paired"][metric]["relative_delta"]
        for metric in ("legalized_hard_cost", "legalized_v_rel")
    }
    checks = {
        "same_parameters": (
            training_summary(args.baseline_dir)["parameters"]
            == training_summary(args.candidate_dir)["parameters"]
        ),
        "dev_non_inferior": (
            training_summary(args.candidate_dir)["dev_total"]
            <= training_summary(args.baseline_dir)["dev_total"]
        ),
        "refined_cost_improves_5pct": (
            relative["legalized_hard_cost"] <= -0.05
        ),
        "refined_v_rel_improves_5pct": (
            relative["legalized_v_rel"] <= -0.05
        ),
    }
    report = {
        "status": "pass" if all(checks.values()) else "reject_path_consistency",
        "scope": "seed-19 5k two-step sampling-path consistency screen",
        "baseline": training_summary(args.baseline_dir),
        "candidate": training_summary(args.candidate_dir),
        "endpoints": endpoints,
        "relative_refined_delta": relative,
        "checks": checks,
        "decision": (
            "promote path consistency"
            if all(checks.values())
            else "retain the established objective and investigate refiner-guided target construction"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], **checks}, indent=2))


if __name__ == "__main__":
    main()
