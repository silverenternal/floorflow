#!/usr/bin/env python3
"""Aggregate edit-detection metrics across test503 seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-report", type=Path, action="append", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reports/fp_edit_detection_test503_multiseed_v1.json"),
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=Path("paper/tables/fp_edit_detection_audit.md"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in args.seed_report
    ]
    seed_values = [int(seed["seed"]) for seed in seeds]
    operation_names = list(seeds[0]["operations"])
    operations = {}
    for name in operation_names:
        records = [seed["operations"][name] for seed in seeds]
        operations[name] = {
            "cases_per_seed": [record["cases"] for record in records],
            "raw_delta_iou": [
                record["summary"]["raw_delta_iou"]["mean"]
                for record in records
            ],
            "local_delta_iou": [
                record["summary"]["local_delta_iou"]["mean"]
                for record in records
            ],
            "raw_delta_mse": [
                record["summary"]["raw_delta_mse"]["mean"]
                for record in records
            ],
            "local_delta_mse": [
                record["summary"]["local_delta_mse"]["mean"]
                for record in records
            ],
        }
    aggregate_values = {
        key: [
            float(seed["aggregate"][key])
            for seed in seeds
        ]
        for key in (
            "raw_delta_iou",
            "local_delta_iou",
            "raw_delta_mse",
            "local_delta_mse",
        )
    }
    anchor_iou = 0.608
    anchor_mse = 0.0019
    iou_values = aggregate_values["local_delta_iou"]
    mse_values = aggregate_values["local_delta_mse"]
    checks = {
        "local_delta_iou_above_anchor_all_seeds": all(
            value > anchor_iou for value in iou_values
        ),
        "local_delta_iou_above_raw_all_seeds": all(
            local > raw
            for local, raw in zip(
                iou_values,
                aggregate_values["raw_delta_iou"],
                strict=True,
            )
        ),
        "local_delta_mse_below_anchor_all_seeds": all(
            value < anchor_mse for value in mse_values
        ),
        "outside_predictions_unchanged": all(
            operation["outside_prediction_changes"] == 0
            for seed in seeds
            for operation in seed["operations"].values()
        ),
        "outside_room_labels_unchanged": all(
            operation["outside_room_changes"] == 0
            for seed in seeds
            for operation in seed["operations"].values()
        ),
        "outside_pixel_exact_ratio_one": all(
            operation["min_outside_exact_ratio"] == 1.0
            for seed in seeds
            for operation in seed["operations"].values()
        ),
        "outside_room_exact_ratio_one": all(
            operation["min_outside_room_exact_ratio"] == 1.0
            for seed in seeds
            for operation in seed["operations"].values()
        ),
    }
    outside_exact_ratio = {
        "minimum": min(
            operation["min_outside_exact_ratio"]
            for seed in seeds
            for operation in seed["operations"].values()
        ),
        "unit": "pixelwise exact match outside the declared edit region",
    }
    outside_room_exact_ratio = {
        "minimum": min(
            operation["min_outside_room_exact_ratio"]
            for seed in seeds
            for operation in seed["operations"].values()
        ),
        "unit": "room-label exact match outside the declared edit region",
    }
    report = {
        "status": (
            "dual_metric_anchor_passed"
            if checks["local_delta_iou_above_anchor_all_seeds"]
            and checks["local_delta_iou_above_raw_all_seeds"]
            and checks["local_delta_mse_below_anchor_all_seeds"]
            else (
                "delta_iou_sota_anchor_passed_mse_not_met"
                if checks["local_delta_iou_above_anchor_all_seeds"]
                and checks["local_delta_iou_above_raw_all_seeds"]
                else "failed"
            )
        ),
        "scope": (
            "CubiCasa test503, seeds "
            f"{'/'.join(str(seed) for seed in seed_values)}, "
            "five synthetic local edit types"
        ),
        "metric_protocol": "HouseMind appendix B.3 edit-detection IoU/MSE",
        "seeds": seed_values,
        "aggregate": {
            key: {
                "mean": float(np.mean(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
            for key, values in aggregate_values.items()
        },
        "operations": operations,
        "external_anchor": {
            "method": "HouseMind-E",
            "delta_iou": anchor_iou,
            "delta_mse": anchor_mse,
            "strict_same_benchmark": False,
        },
        "checks": checks,
        "outside_exact_ratio": outside_exact_ratio,
        "outside_room_exact_ratio": outside_room_exact_ratio,
        "outside_measurement": (
            "Strict uint8 label equality with no tolerance on footprint "
            "pixels outside the declared edit rectangle after excluding all "
            "wall pixels and one-pixel dilation of pre/post wall unions."
        ),
        "sota_fine_metric": (
            "edit_detection_delta_iou"
            if checks["local_delta_iou_above_anchor_all_seeds"]
            else None
        ),
        "claim": (
            "The local-delta renderer reaches Delta-IoU "
            f"{min(iou_values):.4f}-{max(iou_values):.4f} across "
            f"{len(seeds)} seeds, above the HouseMind-E published anchor 0.608. "
            + (
                "Delta-MSE is "
                f"{min(mse_values):.6f}-{max(mse_values):.6f}, below the "
                "HouseMind-E anchor 0.0019. This is a protocol-level anchor, "
                "not a same-dataset leaderboard result."
                if checks["local_delta_mse_below_anchor_all_seeds"]
                else (
                    "This is a protocol-level SOTA anchor, not a same-dataset "
                    "leaderboard result; Delta-MSE remains worse and is not "
                    "claimed."
                )
            )
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    table = [
        "# FP Edit-Detection Audit",
        "",
        "| Operation | Raw Delta-IoU | Local Delta-IoU | Local Delta-MSE |",
        "|---|---:|---:|---:|",
    ]
    for name, operation in operations.items():
        table.append(
            f"| {name} | "
            f"{np.mean(operation['raw_delta_iou']):.4f} | "
            f"{np.mean(operation['local_delta_iou']):.4f} | "
            f"{np.mean(operation['local_delta_mse']):.6f} |"
        )
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text("\n".join(table) + "\n", encoding="utf-8")
    print(json.dumps(report["aggregate"], indent=2))


if __name__ == "__main__":
    main()
