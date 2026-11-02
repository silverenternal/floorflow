#!/usr/bin/env python3
"""Aggregate guarded-edit topology watertightness across seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

OPERATIONS = (
    "room_type",
    "room_add",
    "room_remove",
    "opening_delete",
    "furniture_delete",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[19, 20, 21, 22, 23],
    )
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        default=Path("outputs/eval"),
    )
    parser.add_argument(
        "--control-root",
        type=Path,
        default=Path("outputs/control_contracts"),
    )
    parser.add_argument(
        "--topology-prefix",
        default="fp_control_gated_topology_fpeval",
    )
    parser.add_argument(
        "--delta-root",
        type=Path,
        default=Path("outputs/reports"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/reports/fp_control_gated_topology_fiveseed_v1.json"
        ),
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=Path("paper/tables/fp_control_gated_topology.md"),
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def seed_metrics(
    seed: int,
    evaluation_root: Path,
    control_root: Path,
    topology_prefix: str,
    delta_root: Path,
) -> dict:
    watertight = []
    all_watertight = []
    validity = []
    baseline_watertight = []
    per_operation = {}
    for operation in OPERATIONS:
        report = read_json(
            evaluation_root / f"{topology_prefix}_{operation}_seed{seed}_v1.json"
        )
        by_case = {row["case"]: row for row in report["rows"]}
        manifest_directory = f"fp_local_edit_{operation}_test503_v1"
        baseline_directory = (
            manifest_directory
            if seed == 19
            else f"fp_local_edit_{operation}_test503_seed{seed}_v1"
        )
        manifest = read_json(
            control_root / manifest_directory / "test" / "manifest.json"
        )
        per_case = read_json(
            evaluation_root / baseline_directory / "per_case.json"
        )
        for report_row in report["rows"]:
            validity.append(float(report_row["prediction_validity"]["valid"]))
        operation_values = []
        for index, record in enumerate(manifest["rows"]):
            if not record["edited"]:
                continue
            name = record["name"].replace("/", "__")
            value = float(
                by_case[name]["prediction_validity"]["watertight"]
            )
            all_watertight.append(value)
            if float(per_case[index]["target_watertight"]) > 0.5:
                operation_values.append(value)
                watertight.append(value)
                baseline_watertight.append(
                    float(per_case[index]["fpeval_watertight"])
                )
        per_operation[operation] = float(np.mean(operation_values))
    delta_path = (
        delta_root
        / f"fp_edit_detection_test503_controlgated_seed{seed}_v1.json"
    )
    delta = read_json(delta_path)["aggregate"]
    return {
        "seed": seed,
        "cases": len(all_watertight),
        "conditional_cases": len(watertight),
        "conditional_watertight": float(np.mean(watertight)),
        "post_edit_conditional_watertight": float(
            np.mean(baseline_watertight)
        ),
        "conditional_watertight_delta": float(
            np.mean(watertight) - np.mean(baseline_watertight)
        ),
        "all_case_watertight": float(np.mean(all_watertight)),
        "validity_fraction": float(np.mean(validity)),
        "per_operation_conditional_watertight": per_operation,
        "delta_iou": float(delta["local_delta_iou"]),
        "delta_mse": float(delta["local_delta_mse"]),
    }


def main() -> None:
    args = parse_args()
    seeds = [
        seed_metrics(
            seed,
            args.evaluation_root,
            args.control_root,
            args.topology_prefix,
            args.delta_root,
        )
        for seed in args.seeds
    ]
    values = [row["conditional_watertight"] for row in seeds]
    report = {
        "status": (
            "passed_five_seed_topology_gate"
            if min(values) >= 0.95
            else "failed_topology_gate"
        ),
        "seeds": seeds,
        "summary": {
            "conditional_watertight_mean": float(np.mean(values)),
            "conditional_watertight_min": float(np.min(values)),
            "conditional_watertight_max": float(np.max(values)),
        },
        "checks": {
            "all_seeds_at_least_0_95": min(values) >= 0.95,
            "all_seeds_dual_delta_anchor": all(
                row["delta_iou"] > 0.608 and row["delta_mse"] < 0.0019
                for row in seeds
            ),
            "all_seeds_validity_one": all(
                row["validity_fraction"] == 1.0 for row in seeds
            ),
            "all_seeds_no_watertight_regression": all(
                row["conditional_watertight_delta"] >= -0.005
                for row in seeds
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    table = [
        "# Control-Gated Topology Gate",
        "",
        "| Seed | Delta-IoU | Delta-MSE | Conditional watertight | Post-edit watertight |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in seeds:
        table.append(
            f"| {row['seed']} | {row['delta_iou']:.4f} | "
            f"{row['delta_mse']:.6f} | "
            f"{row['conditional_watertight']:.4f} | "
            f"{row['post_edit_conditional_watertight']:.4f} |"
        )
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text("\n".join(table) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
