#!/usr/bin/env python3
"""Compare the velocity and x0/dropout inference ensemble."""

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
        "--ensemble-dir",
        type=Path,
        default=ROOT
        / "outputs/chip_global_rescue/screen_5k/"
        "graph_ensemble_velocity_x0drop_seed19_5k",
    )
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_inference_ensemble_screen_v1.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    endpoints = {
        "refined": {
            "baseline": endpoint_summary(args.baseline_dir, "refined"),
            "ensemble": endpoint_summary(args.ensemble_dir, "refined"),
            "paired": paired_metrics(
                endpoint_rows(args.baseline_dir, "refined"),
                endpoint_rows(args.ensemble_dir, "refined"),
                iterations=args.bootstrap,
            ),
        }
    }
    refined = {
        metric: endpoints["refined"]["paired"][metric]["relative_delta"]
        for metric in ("legalized_hard_cost", "legalized_v_rel")
    }
    checks = {
        "refined_cost_improves": refined["legalized_hard_cost"] < 0.0,
        "refined_v_rel_improves": refined["legalized_v_rel"] < 0.0,
        "refined_both_improve_5pct": all(value <= -0.05 for value in refined.values()),
    }
    report = {
        "status": "keep_single_model" if not checks["refined_both_improve_5pct"] else "promote_ensemble",
        "scope": "equal-weight inference ensemble of velocity full graph and 5k x0+dropout",
        "endpoints": endpoints,
        "relative_refined_delta": refined,
        "checks": checks,
        "decision": (
            "keep the single velocity model"
            if not checks["refined_both_improve_5pct"]
            else "train the x0+dropout member to the full budget"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], **checks}, indent=2))


if __name__ == "__main__":
    main()
