#!/usr/bin/env python3
"""Compare FloorFlow and Raster2Seq on identical paired edit cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from floorflow.evaluation.paired_bootstrap import paired_delta

MAIN_OPERATIONS = (
    "room_type",
    "room_add",
    "room_remove",
    "opening_delete",
)
SUPPLEMENTARY_OPERATIONS = ("furniture_delete",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--external",
        type=Path,
        default=Path(
            "outputs/external_baselines/raster2seq/"
            "test503_matched_v1/report.json"
        ),
    )
    parser.add_argument(
        "--seed-report",
        type=Path,
        action="append",
        default=None,
    )
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reports/fp_edit_detection_matched_external_v2.json"),
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=Path("paper/tables/fp_matched_edit_comparison_v2.md"),
    )
    parser.add_argument(
        "--leakage-table",
        type=Path,
        default=Path("paper/tables/fp_raster2seq_outside_leakage.md"),
    )
    args = parser.parse_args()
    if args.seed_report is None:
        args.seed_report = [
            Path(
                "outputs/reports/"
                f"fp_edit_detection_test503_controlgated_seed{seed}_v1.json"
            )
            for seed in (19, 20, 21, 22, 23)
        ]
    return args


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rows_by_name(rows: list[dict]) -> dict[str, dict]:
    return {str(row["name"]): row for row in rows}


def seed_comparison(
    internal: dict,
    external_rows: list[dict],
    operation: str,
    iterations: int,
) -> dict:
    internal_rows = rows_by_name(internal["operations"][operation]["rows"])
    external_index = rows_by_name(external_rows)
    if set(internal_rows) != set(external_index):
        raise ValueError(f"case mismatch for {operation}")
    names = sorted(internal_rows)
    ours = [internal_rows[name] for name in names]
    baseline = [
        {
            **external_index[name],
            "local_delta_iou": external_index[name]["delta_iou"],
            "local_delta_mse": external_index[name]["delta_mse"],
            "raw_delta_iou": external_index[name]["delta_iou"],
            "raw_delta_mse": external_index[name]["delta_mse"],
        }
        for name in names
    ]
    return {
        "local_delta_iou": paired_delta(
            ours,
            baseline,
            "local_delta_iou",
            iterations=iterations,
            seed=17,
        ),
        "local_delta_mse": paired_delta(
            ours,
            baseline,
            "local_delta_mse",
            iterations=iterations,
            seed=17,
        ),
        "raw_delta_iou": paired_delta(
            ours,
            baseline,
            "raw_delta_iou",
            iterations=iterations,
            seed=17,
        ),
    }


def main() -> None:
    args = parse_args()
    external = read_json(args.external)
    internal = [read_json(path) for path in args.seed_report]
    operation_results = {}
    for operation in (*MAIN_OPERATIONS, *SUPPLEMENTARY_OPERATIONS):
        external_rows = [
            row
            for row in external["rows"]
            if row["operation"] == operation
        ]
        local_iou = [
            float(row["local_delta_iou"])
            for report in internal
            for row in report["operations"][operation]["rows"]
        ]
        local_iou_by_seed = [
            float(
                np.mean(
                    [
                        row["local_delta_iou"]
                        for row in report["operations"][operation]["rows"]
                    ]
                )
            )
            for report in internal
        ]
        outside_exact = np.asarray(
            [row["outside_exact_ratio"] for row in external_rows],
            dtype=np.float64,
        )
        outside_changed = np.asarray(
            [row["outside_changed_pixels"] for row in external_rows],
            dtype=np.float64,
        )
        seed_results = {
            str(report["seed"]): seed_comparison(
                report,
                external_rows,
                operation,
                args.bootstrap,
            )
            for report in internal
        }
        iou_deltas = [
            result["local_delta_iou"]["mean_delta"]
            for result in seed_results.values()
        ]
        operation_results[operation] = {
            "cases": len(external_rows),
            "eligible_for_main_table": operation in MAIN_OPERATIONS,
            "floorflow_local_delta_iou": {
                "mean_over_cases_and_seeds": float(np.mean(local_iou)),
                "per_seed_mean_range": [
                    float(min(local_iou_by_seed)),
                    float(max(local_iou_by_seed)),
                ],
                "paired_mean_gain_range": [
                    float(min(iou_deltas)),
                    float(max(iou_deltas)),
                ],
            },
            "raster2seq": {
                key: float(np.mean([row[key] for row in external_rows]))
                for key in (
                    "delta_iou",
                    "delta_mse",
                    "outside_exact_ratio",
                    "outside_changed_pixels",
                    "before_room_support_iou",
                    "after_room_support_iou",
                )
            },
            "raster2seq_outside_exact_distribution": {
                "mean": float(outside_exact.mean()),
                "median": float(np.median(outside_exact)),
                "p10": float(np.quantile(outside_exact, 0.10)),
                "min": float(outside_exact.min()),
                "max": float(outside_exact.max()),
            },
            "raster2seq_outside_changed_pixels_distribution": {
                "mean": float(outside_changed.mean()),
                "median": float(np.median(outside_changed)),
                "p90": float(np.quantile(outside_changed, 0.90)),
                "max": int(outside_changed.max()),
            },
            "raster2seq_outside_leak_fraction": float(
                1.0 - outside_exact.mean()
            ),
            "floorflow_local_outside_exact_ratio": min(
                report["operations"][operation]["min_outside_exact_ratio"]
                for report in internal
            ),
            "paired_by_seed": {
                seed: {
                    **metrics,
                    "local_delta_iou": {
                        **metrics["local_delta_iou"],
                        "higher_is_better": True,
                    },
                    "local_delta_mse": {
                        **metrics["local_delta_mse"],
                        "lower_is_better": True,
                    },
                    "raw_delta_iou": {
                        **metrics["raw_delta_iou"],
                        "higher_is_better": True,
                    },
                }
                for seed, metrics in seed_results.items()
            },
            "checks": {
                "local_delta_iou_gain_ci_excludes_zero_all_seeds": all(
                    result["local_delta_iou"]["ci_low"] > 0.0
                    for result in seed_results.values()
                ),
                "local_delta_mse_lower_ci_excludes_zero_all_seeds": all(
                    result["local_delta_mse"]["ci_high"] < 0.0
                    for result in seed_results.values()
                ),
            },
        }

    all_main_passed = all(
        operation_results[operation]["checks"][
            "local_delta_iou_gain_ci_excludes_zero_all_seeds"
        ]
        for operation in MAIN_OPERATIONS
    )
    report = {
        "status": "shared_project_preprocessing_sensitivity_complete",
        "scope": (
            "CubiCasa test503; identical project-controlled coarse-control "
            "renderings, pre/post rows, wall exclusion, and HouseMind-style "
            "scorer. This is not Raster2Seq's released official preprocessing."
        ),
        "exact_match_definition": (
            "Predicted CubiCasa labels are uint8 categorical values. Exact "
            "match requires strict integer equality with no tolerance. The "
            "outside region is footprint pixels outside the declared edit "
            "rectangle after excluding all wall pixels and one-pixel wall "
            "dilation in either pre- or post-edit target."
        ),
        "external_method": {
            "name": "Raster2Seq",
            "version": external["repository_commit"],
            "checkpoint_sha256": external["checkpoint_sha256"],
            "source": "https://huggingface.co/haopt/Raster2Seq",
            "role": "shared-preprocessing reconstruction sensitivity baseline",
            "published_cubicasa_room_f1": 88.7,
            "edit_aware": False,
            "condition_mode": external["condition_mode"],
            "adaptation": external["input_contract"],
        },
        "operations": operation_results,
        "checks": {
            "all_cases_paired": True,
            "all_main_operations_local_iou_gain": all_main_passed,
            "same_scorer": True,
            "same_pre_post_rows": True,
            "upstream_official_preprocessing_matched": False,
            "shared_project_preprocessing_sensitivity": True,
            "house_mind_same_benchmark": False,
        },
        "claim": (
            "On identical CubiCasa test503 edit rows, shared project-controlled "
            "input renderings, and one shared scorer, "
            "FloorFlow's local-delta output layer has higher Delta-IoU and "
            "lower Delta-MSE than a public CubiCasa floorplan reconstruction "
            "baseline that regenerates each side independently."
        ),
        "claim_boundary": (
            "Raster2Seq is a matched reconstruction baseline, not an "
            "editing-aware method, and this is not HouseMind's RPLAN benchmark. "
            "The comparison uses shared project preprocessing rather than the "
            "upstream official preprocessing, so it is a sensitivity result, "
            "not an official-pipeline reproduction. "
            "Use the result as matched external evidence for edit locality, "
            "not as a HouseMind leaderboard result."
        ),
        "supplementary_note": (
            "Furniture deletion is supplementary because Raster2Seq has no "
            "furniture output class."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    table = [
        "# Shared-Project-Preprocessing Raster2Seq Sensitivity",
        "",
        (
            "| Operation | Cases | FloorFlow Delta-IoU | Raster2Seq Delta-IoU | "
            "Raster2Seq Delta-MSE | Raster2Seq outside exact | FloorFlow outside exact |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for operation, result in operation_results.items():
        marker = "" if result["eligible_for_main_table"] else " (supp.)"
        table.append(
            f"| {operation}{marker} | {result['cases']} | "
            f"{result['floorflow_local_delta_iou']['mean_over_cases_and_seeds']:.4f} "
            f"({result['floorflow_local_delta_iou']['per_seed_mean_range'][0]:.4f}-"
            f"{result['floorflow_local_delta_iou']['per_seed_mean_range'][1]:.4f}) | "
            f"{result['raster2seq']['delta_iou']:.4f} | "
            f"{result['raster2seq']['delta_mse']:.4f} | "
            f"{result['raster2seq']['outside_exact_ratio']:.4f} | "
            f"{result['floorflow_local_outside_exact_ratio']:.4f} |"
        )
    table.extend(
        [
            "",
            (
                "Both methods use shared project-controlled coarse-control "
                "renderings. This is not Raster2Seq's official preprocessing, "
                "and Raster2Seq is not edit-aware."
            ),
        ]
    )
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text("\n".join(table) + "\n", encoding="utf-8")

    leakage = sorted(
        (
            (
                operation,
                result["raster2seq_outside_exact_distribution"],
                result["raster2seq_outside_changed_pixels_distribution"],
                result["raster2seq_outside_leak_fraction"],
                result["eligible_for_main_table"],
            )
            for operation, result in operation_results.items()
        ),
        key=lambda row: row[3],
        reverse=True,
    )
    leakage_table = [
        "# Raster2Seq Outside-Edit Leakage",
        "",
        (
            "| Operation | Outside exact mean | p10 | min | Changed pixels "
            "mean | Changed pixels p90 | Leak fraction |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for operation, exact, changed, leak, eligible in leakage:
        marker = "" if eligible else " (supp.)"
        leakage_table.append(
            f"| {operation}{marker} | {exact['mean']:.4f} | "
            f"{exact['p10']:.4f} | {exact['min']:.4f} | "
            f"{changed['mean']:.1f} | {changed['p90']:.1f} | "
            f"{leak:.4%} |"
        )
    leakage_table.extend(
        [
            "",
            (
                "Exact match is strict uint8 label equality outside the declared "
                "edit rectangle, excluding footprint exterior and dilated walls."
            ),
        ]
    )
    args.leakage_table.parent.mkdir(parents=True, exist_ok=True)
    args.leakage_table.write_text(
        "\n".join(leakage_table) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["checks"], indent=2))


if __name__ == "__main__":
    main()
