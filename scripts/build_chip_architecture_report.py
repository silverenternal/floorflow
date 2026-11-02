#!/usr/bin/env python3
"""Compare chip output parameterization and conditioning dropout screens."""

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

ARMS = (
    "x0_seed19_5k",
    "dropout_seed19_5k",
    "x0_dropout_seed19_5k",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT / "outputs/chip_architecture_v1/screen_5k",
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=ROOT
        / "outputs/chip_graph_mechanism/screen_5k/graph_full_seed19_5k",
    )
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "outputs/reports/chip_output_parameterization_and_dropout_v1.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline_training = read(
        args.baseline_dir.parent / f"{args.baseline_dir.name}.json"
    )
    baseline = {
        "parameters": int(baseline_training["model_parameters"]),
        "dev_total": float(baseline_training["final_dev"]["total"]),
        "raw": endpoint_summary(args.baseline_dir, "raw"),
        "refined": endpoint_summary(args.baseline_dir, "refined"),
    }
    arms = {}
    for arm in ARMS:
        arm_dir = args.root / arm
        training = read(args.root / f"{arm}.json")
        summary = {
            "parameters": int(training["model_parameters"]),
            "dev_total": float(training["final_dev"]["total"]),
            "raw": endpoint_summary(arm_dir, "raw"),
            "refined": endpoint_summary(arm_dir, "refined"),
            "paired_vs_baseline": {
                kind: paired_metrics(
                    endpoint_rows(args.baseline_dir, kind),
                    endpoint_rows(arm_dir, kind),
                    iterations=args.bootstrap,
                )
                for kind in ("raw", "refined")
            },
        }
        summary["relative_refined_delta"] = {
            metric: (
                summary["refined"][metric] - baseline["refined"][metric]
            )
            / baseline["refined"][metric]
            for metric in ("legalized_hard_cost", "legalized_v_rel")
        }
        arms[arm] = summary

    qualifying = [
        arm
        for arm, row in arms.items()
        if all(value <= -0.05 for value in row["relative_refined_delta"].values())
    ]
    winner = (
        min(
            qualifying,
            key=lambda arm: (
                arms[arm]["refined"]["legalized_hard_cost"]
                + arms[arm]["refined"]["legalized_v_rel"]
            ),
        )
        if qualifying
        else None
    )
    report = {
        "status": "promote" if winner is not None else "keep_velocity_full_graph",
        "scope": "seed-19 5k output-parameterization and conditioning-dropout screens",
        "root": str(args.root),
        "baseline": baseline,
        "arms": arms,
        "gate": "both refined primary metrics improve at least 5%",
        "winner": winner,
        "decision": (
            f"promote {winner} to warm-restart and seed replication"
            if winner is not None
            else "retain the full-graph velocity model"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "winner": winner}, indent=2))


if __name__ == "__main__":
    main()
