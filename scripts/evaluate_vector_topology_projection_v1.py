#!/usr/bin/env python3
"""Evaluate edit-local topology projection on hardened prediction rasters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.fevp_metric_contracts import (
    load_metric_contract,
    tolerances_from_contract,
    validity_checks,
    watertight_diagnostics,
)
from floorflow.evaluation.structural_legality import structural_legality_metrics
from floorflow.evaluation.typed_edit_metrics import normalized_region_mask
from floorflow.evaluation.vector_edit_protocol import vector_preservation_metrics
from floorflow.geometry.wall_graph_readout import raster_labels_to_wall_plan
from floorflow.inference.vector_topology_projection import project_edit_topology

ROOT = Path(__file__).resolve().parents[1]
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
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--skip-hole-fill", action="store_true")
    parser.add_argument("--skip-wall-projection", action="store_true")
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=ROOT / "outputs/cache/cubicasa5k_full_v2/taxonomy.json",
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs/fevp_metric_contract_v2.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/vector_topology_projection_v1.json",
    )
    return parser.parse_args()


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as payload:
        return {key: payload[key] for key in payload.files}


def _tasks(seed: int, limit: int) -> list[dict]:
    tasks = []
    for operation in OPERATIONS:
        manifest = json.loads(
            (
                ROOT
                / f"outputs/control_contracts/fp_local_edit_{operation}_test503_v1"
                / "test/manifest.json"
            ).read_text(encoding="utf-8")
        )
        for record in manifest["rows"]:
            if record["edited"]:
                tasks.append({"operation": operation, "record": record})
    tasks.sort(key=lambda row: (row["operation"], row["record"]["name"]))
    quota = max(1, int(limit) // len(OPERATIONS))
    selected = []
    for operation in OPERATIONS:
        selected.extend(
            [row for row in tasks if row["operation"] == operation][:quota]
        )
    return selected[: int(limit)]


def _topology(labels, room_labels, footprint, taxonomy, tolerances) -> dict:
    plan = raster_labels_to_wall_plan(
        labels,
        taxonomy,
        room_labels=room_labels,
    )
    validity = validity_checks(
        plan,
        footprint,
        taxonomy,
        tolerances=tolerances,
        room_labels=room_labels,
    )
    watertight = watertight_diagnostics(
        plan,
        taxonomy=taxonomy,
        labels=labels,
        footprint=footprint,
        fpeval_watertight=False,
        tolerances=tolerances,
    )
    return {
        "validity": validity,
        "watertight": watertight,
    }


def main() -> None:
    args = parse_args()
    taxonomy = FloorplanTaxonomy.read(args.taxonomy)
    contract = load_metric_contract(args.contract)
    tolerances = tolerances_from_contract(contract)
    rows = []
    for task in _tasks(args.seed, args.limit):
        operation = task["operation"]
        record = task["record"]
        safe_name = str(record["name"]).replace("/", "__")
        before_path = (
            ROOT
            / f"outputs/eval/fp_hard_metrics_full_test503_seed{args.seed}_v1"
            / "rasters"
            / f"{safe_name}.npz"
        )
        after_path = (
            ROOT
            / f"outputs/eval/fp_control_gated_local_test503_seed{args.seed}_v1"
            / operation
            / f"{safe_name}.npz"
        )
        before = _load(before_path)
        after = _load(after_path)
        footprint = np.asarray(after["footprint"], dtype=bool)
        region = normalized_region_mask(
            footprint.shape,
            record["edit_region_normalized"],
        ) & footprint
        projected = project_edit_topology(
            before["prediction"],
            after["prediction"],
            after["room_labels"],
            region,
            footprint,
            taxonomy,
            fill_holes=not args.skip_hole_fill,
            project_walls=not args.skip_wall_projection,
        )
        baseline_vector = vector_preservation_metrics(
            before["prediction"],
            before["room_labels"],
            after["prediction"],
            after["room_labels"],
            region,
            taxonomy,
        )
        projected_vector = vector_preservation_metrics(
            before["prediction"],
            before["room_labels"],
            projected.labels,
            projected.room_labels,
            region,
            taxonomy,
        )
        baseline_topology = _topology(
            after["prediction"],
            after["room_labels"],
            footprint,
            taxonomy,
            tolerances,
        )
        projected_topology = _topology(
            projected.labels,
            projected.room_labels,
            footprint,
            taxonomy,
            tolerances,
        )
        baseline_graph = structural_legality_metrics(
            after["prediction"],
            after["target"],
            taxonomy,
        )
        projected_graph = structural_legality_metrics(
            projected.labels,
            after["target"],
            taxonomy,
        )
        rows.append(
            {
                "case_id": f"seed{args.seed}/{operation}/{record['name']}",
                "operation": operation,
                "outside_changed_pixels": projected.outside_changed_pixels,
                "changed_pixels": projected.changed_pixels,
                "filled_room_pixels": projected.filled_room_pixels,
                "projected_wall_pixels": projected.projected_wall_pixels,
                "baseline_vector": baseline_vector,
                "projected_vector": projected_vector,
                "baseline_topology": baseline_topology,
                "projected_topology": projected_topology,
                "baseline_graph": baseline_graph,
                "projected_graph": projected_graph,
            }
        )
    summary = {
        "cases": len(rows),
        "outside_exact_cases": sum(
            row["outside_changed_pixels"] == 0 for row in rows
        ),
        "projected_wall_pixels": sum(
            row["projected_wall_pixels"] for row in rows
        ),
        "baseline_validity": float(
            np.mean(
                [row["baseline_topology"]["validity"]["validity"] for row in rows]
            )
        ),
        "projected_validity": float(
            np.mean(
                [row["projected_topology"]["validity"]["validity"] for row in rows]
            )
        ),
        "baseline_watertight": float(
            np.mean(
                [
                    row["baseline_topology"]["watertight"]["watertight_rate"]
                    for row in rows
                ]
            )
        ),
        "projected_watertight": float(
            np.mean(
                [
                    row["projected_topology"]["watertight"]["watertight_rate"]
                    for row in rows
                ]
            )
        ),
        "baseline_node_f1": float(
            np.mean([row["baseline_graph"]["node_f1"] for row in rows])
        ),
        "projected_node_f1": float(
            np.mean([row["projected_graph"]["node_f1"] for row in rows])
        ),
    }
    report = {
        "status": "complete",
        "seed": args.seed,
        "variant": {
            "fill_holes": not args.skip_hole_fill,
            "project_walls": not args.skip_wall_projection,
        },
        "summary": summary,
        "checks": {
            "outside_exact_all_cases": (
                summary["outside_exact_cases"] == len(rows)
            ),
            "projected_validity_not_lower": (
                summary["projected_validity"]
                >= summary["baseline_validity"] - 1.0e-12
            ),
            "projected_watertight_not_lower": (
                summary["projected_watertight"]
                >= summary["baseline_watertight"] - 1.0e-12
            ),
            "node_f1_not_lower": (
                summary["projected_node_f1"]
                >= summary["baseline_node_f1"] - 1.0e-12
            ),
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"] | {"checks": report["checks"]}, indent=2))
    raise SystemExit(0 if all(report["checks"].values()) else 1)


__all__ = ["main"]


if __name__ == "__main__":
    main()
