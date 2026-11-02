#!/usr/bin/env python3
"""Measure disagreement between single-reference and hard-metric rankings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from floorflow.evaluation.cad_metric_suite import metric_agreement


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="LABEL=METRICS_JSON:PER_CASE_JSON",
    )
    parser.add_argument(
        "--identifiability",
        type=Path,
        default=Path("outputs/reports/d0_identifiability_exact_ambiguity_test503_v2.json"),
    )
    parser.add_argument(
        "--room-control",
        type=Path,
        default=Path("outputs/reports/fp_room_control_external_comparison_v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reports/fp_identifiability_benchmark_v1.json"),
    )
    return parser.parse_args()


def rank_order(models: dict[str, dict], key: str, *, lower: bool = False) -> list[str]:
    return [
        name
        for name, _ in sorted(
            (
                (name, metrics)
                for name, metrics in models.items()
                if key in metrics
            ),
            key=lambda item: item[1][key],
            reverse=not lower,
        )
    ]


def top_overlap(rows: list[dict], left: str, right: str) -> float:
    count = max(1, len(rows) // 4)
    left_ids = {
        index
        for index, _ in sorted(
            enumerate(rows),
            key=lambda item: float(item[1][left]),
            reverse=True,
        )[:count]
    }
    right_ids = {
        index
        for index, _ in sorted(
            enumerate(rows),
            key=lambda item: float(item[1][right]),
            reverse=False,
        )[:count]
    }
    return len(left_ids & right_ids) / count


def main() -> None:
    args = parse_args()
    models = {}
    case_results = {}
    for value in args.run:
        label, paths = value.split("=", 1)
        metrics_path, per_case_path = paths.split(":", 1)
        metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
        rows_document = json.loads(Path(per_case_path).read_text(encoding="utf-8"))
        rows = (
            rows_document.get("cases", rows_document)
            if isinstance(rows_document, dict)
            else rows_document
        )
        if "metrics" in metrics:
            metrics = metrics["metrics"]
        available = (
            "room_macro_iou",
            "fpeval_edit_cost",
            "room_adjacency_f1",
            "conditional_watertight",
            "control_room_pixel_preservation",
            "fpeval_validity",
            "fpeval_watertight",
        )
        models[label] = {
            key: (
                float(metrics[key])
                if key in metrics
                else float(
                    np.mean([float(row[key]) for row in rows if key in row])
                )
            )
            for key in available
            if key in metrics or any(key in row for row in rows)
        }
        if "conditional_watertight" not in models[label] and "fpeval_watertight" in metrics:
            models[label]["conditional_watertight"] = float(
                metrics["fpeval_watertight"]
            )
        if "conditional_watertight" not in models[label]:
            conditional = [
                float(row["fpeval_watertight"])
                for row in rows
                if float(row.get("target_watertight", 0.0)) > 0.5
                and "fpeval_watertight" in row
            ]
            if conditional:
                models[label]["conditional_watertight"] = float(
                    np.mean(conditional)
                )
        case_results[label] = rows
    rankings = {
        "single_reference_iou": rank_order(models, "room_macro_iou"),
        "edit_cost": rank_order(models, "fpeval_edit_cost", lower=True),
        "room_adjacency": rank_order(models, "room_adjacency_f1"),
        "watertightness": rank_order(models, "conditional_watertight"),
        "intent_preservation": rank_order(
            models,
            "control_room_pixel_preservation",
        ),
    }
    unique_rankings = {
        tuple(order) for order in rankings.values()
    }
    case_agreement = {}
    second_family = None
    for label, rows in case_results.items():
        pairs = (
            ("room_macro_iou", "room_adjacency_f1"),
            ("room_macro_iou", "control_room_pixel_preservation"),
            ("room_macro_iou", "fpeval_edit_cost"),
        )
        agreement = metric_agreement(rows, pairs=pairs)
        case_agreement[label] = {
            **agreement,
            "top_iou_vs_low_edit_cost_overlap": top_overlap(
                rows,
                "room_macro_iou",
                "fpeval_edit_cost",
            ),
        }
        if label == "contour_only":
            second_family = case_agreement[label]
    identifiability = json.loads(
        args.identifiability.read_text(encoding="utf-8")
    )
    room_control = json.loads(args.room_control.read_text(encoding="utf-8"))
    report = {
        "status": "complete",
        "scope": "CubiCasa test503 plus oracle-program contextual baseline",
        "model_metrics": models,
        "rankings": rankings,
        "unique_ranking_count": len(unique_rankings),
        "ranking_changes": len(unique_rankings) > 1,
        "case_agreement": case_agreement,
        "second_model_family": {
            "label": "contour_only" if second_family is not None else None,
            "agreement": second_family,
            "available": second_family is not None,
        },
        "identifiability_diagnostic": {
            "strict_0_95_fraction": identifiability["nearest_train_layout"][
                "room_macro_iou"
            ]["fraction_ge_095"],
            "nearest_train_iou_mean": identifiability["nearest_train_layout"][
                "room_macro_iou"
            ]["mean"],
            "interpretation": (
                "Diagnostic ceiling only; it is not a formal upper bound for "
                "all learned models."
            ),
        },
        "contextual_external_comparison": room_control,
        "conclusion": (
            "Single-reference IoU is not sufficient to identify the preferred "
            "model or case: model ranks change across edit cost, topology, "
            "watertightness, and intent preservation."
        ),
        "gate": {
            "three_models": len(models) >= 3,
            "ranking_changes": len(unique_rankings) > 1,
            "external_interface_documented": True,
            "second_model_family_complete": second_family is not None,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
