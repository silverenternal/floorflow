#!/usr/bin/env python3
"""Measure supplementary structural-legality metrics for control-gated edits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.structural_legality import structural_legality_metrics

OPERATIONS = (
    "room_type",
    "room_add",
    "room_remove",
    "opening_delete",
    "furniture_delete",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=Path("outputs/cache/cubicasa5k_full_v2/taxonomy.json"),
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
        "--output",
        type=Path,
        default=Path("outputs/reports/fp_control_gated_structural_seed19_v1.json"),
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=Path("paper/tables/fp_control_gated_structural_metrics.md"),
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    taxonomy = FloorplanTaxonomy.read(args.taxonomy)
    rows = []
    operation_results = {}
    for operation in OPERATIONS:
        manifest = read_json(
            args.control_root
            / f"fp_local_edit_{operation}_test503_v1"
            / "test"
            / "manifest.json"
        )
        fpeval = read_json(
            args.evaluation_root
            / f"fp_control_gated_topology_fpeval_{operation}_seed{args.seed}_v1.json"
        )
        by_case = {row["case"]: row for row in fpeval["rows"]}
        baseline_directory = (
            f"fp_local_edit_{operation}_test503_v1"
            if args.seed == 19
            else f"fp_local_edit_{operation}_test503_seed{args.seed}_v1"
        )
        per_case = read_json(
            args.evaluation_root / baseline_directory / "per_case.json"
        )
        operation_rows = []
        for index, record in enumerate(manifest["rows"]):
            if not record["edited"]:
                continue
            name = record["name"].replace("/", "__")
            with np.load(
                args.evaluation_root
                / f"fp_control_gated_local_test503_seed{args.seed}_v1"
                / operation
                / f"{name}.npz"
            ) as payload:
                prediction = payload["topology_prediction"]
                target = payload["target"]
            metrics = structural_legality_metrics(
                prediction,
                target,
                taxonomy,
            )
            metrics.update(
                {
                    "operation": operation,
                    "name": record["name"],
                    "validity": float(
                        by_case[name]["prediction_validity"]["valid"]
                    ),
                    "conditional_target": bool(
                        per_case[index]["target_watertight"] > 0.5
                    ),
                }
            )
            operation_rows.append(metrics)
            rows.append(metrics)
        operation_results[operation] = {
            key: float(np.mean([row[key] for row in operation_rows]))
            for key in (
                "parsable",
                "validity",
                "room_closure_rate",
                "wall_continuity_rate",
                "self_intersection_rate",
                "room_overlap_ratio",
                "out_of_boundary_ratio",
                "node_f1",
                "edge_overlap",
                "ged",
            )
        }
        print(f"audited {operation}: {len(operation_rows)} cases", flush=True)

    aggregate = {
        key: float(np.mean([row[key] for row in rows]))
        for key in (
            "parsable",
            "validity",
            "room_closure_rate",
            "wall_continuity_rate",
            "self_intersection_rate",
            "room_overlap_ratio",
            "out_of_boundary_ratio",
            "node_f1",
            "edge_overlap",
            "ged",
        )
    }
    aggregate["invalid_rate"] = 1.0 - aggregate["validity"]
    report = {
        "status": "complete",
        "seed": args.seed,
        "cases": len(rows),
        "definitions": {
            "parsable": "wall graph and at least one room polygon are recoverable",
            "room_closure_rate": "fraction of predicted room polygons that are valid",
            "wall_continuity_rate": "fraction of clustered endpoints incident to at least two segments",
            "self_intersection_rate": "fraction of invalid room polygons",
            "room_overlap_ratio": "pairwise room intersection area divided by room union area",
            "out_of_boundary_ratio": "room area outside the image frame divided by room union area",
            "node_f1": "F1 of label-and-IoU-matched room nodes",
            "edge_overlap": "Jaccard overlap of adjacency edges after node matching",
            "ged": "node insertion/deletion plus unmatched adjacency edge count",
        },
        "aggregate": aggregate,
        "operations": operation_results,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    table = [
        f"# Structural Legality Audit (Seed {args.seed})",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    labels = {
        "invalid_rate": "Invalid rate",
        "parsable": "Parsable rate",
        "room_closure_rate": "Room closure rate",
        "wall_continuity_rate": "Wall continuity rate",
        "self_intersection_rate": "Self-intersection rate",
        "room_overlap_ratio": "Room overlap ratio",
        "out_of_boundary_ratio": "Out-of-boundary ratio",
        "node_f1": "Node F1",
        "edge_overlap": "Edge overlap",
        "ged": "GED",
    }
    for key, label in labels.items():
        table.append(f"| {label} | {aggregate[key]:.6f} |")
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text("\n".join(table) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
