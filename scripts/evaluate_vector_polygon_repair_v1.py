#!/usr/bin/env python3
"""Evaluate conservative vector room-polygon repair on 100 edit cases."""

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
)
from floorflow.evaluation.room_graph_matching import (
    build_direct_room_graph,
    build_graph_from_rows,
    match_room_graphs,
)
from floorflow.evaluation.typed_edit_metrics import normalized_region_mask
from floorflow.evaluation.vector_edit_protocol import (
    VectorPlan,
    extract_vector_plan,
    vector_plan_preservation_metrics,
)
from floorflow.geometry.wall_graph_readout import (
    RasterWallPlan,
    ReadoutOpening,
    ReadoutWall,
)
from floorflow.inference.vector_topology_projection import (
    repair_edited_room_polygons,
)

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
        default=ROOT / "outputs/reports/vector_polygon_repair_v1.json",
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


def _wall_plan(plan: VectorPlan) -> RasterWallPlan:
    return RasterWallPlan(
        frame=256.0,
        walls=[
            ReadoutWall(
                start=wall.start,
                end=wall.end,
                thickness=wall.thickness,
                wall_type=wall.wall_type,
            )
            for wall in plan.walls
        ],
        rooms=[
            {
                "label": room.label,
                "walls": [],
                "polygon": np.asarray(room.polygon.exterior.coords).tolist(),
            }
            for room in plan.rooms
        ],
        openings=[
            ReadoutOpening(
                center=opening.center,
                width=opening.width,
                kind=opening.kind,
                wall_index=-1,
            )
            for opening in plan.openings
        ],
    )


def _graph(plan: VectorPlan, room_labels: np.ndarray, taxonomy):
    return build_graph_from_rows(
        [
            {
                "label": room.label,
                "polygon": np.asarray(room.polygon.exterior.coords).tolist(),
            }
            for room in plan.rooms
        ],
        room_labels,
        taxonomy,
        opening_count=len(plan.openings),
    )


def _invalid_affected_rooms(plan: VectorPlan, region: np.ndarray) -> int:
    return sum(
        bool((room.mask & region).any())
        and (
            room.polygon.is_empty
            or not room.polygon.is_valid
            or room.polygon.area <= 0.0
            or len(room.polygon.exterior.coords) < 4
        )
        for room in plan.rooms
    )


def main() -> None:
    args = parse_args()
    taxonomy = FloorplanTaxonomy.read(args.taxonomy)
    tolerances = tolerances_from_contract(load_metric_contract(args.contract))
    rows = []
    for task in _tasks(args.seed, args.limit):
        operation = task["operation"]
        record = task["record"]
        safe_name = str(record["name"]).replace("/", "__")
        before = _load(
            ROOT
            / f"outputs/eval/fp_hard_metrics_full_test503_seed{args.seed}_v1"
            / "rasters"
            / f"{safe_name}.npz"
        )
        after = _load(
            ROOT
            / f"outputs/eval/fp_control_gated_local_test503_seed{args.seed}_v1"
            / operation
            / f"{safe_name}.npz"
        )
        footprint = np.asarray(after["footprint"], dtype=bool)
        region = normalized_region_mask(
            footprint.shape,
            record["edit_region_normalized"],
        ) & footprint
        before_plan = extract_vector_plan(
            before["prediction"],
            before["room_labels"],
            taxonomy,
        )
        after_plan = extract_vector_plan(
            after["prediction"],
            after["room_labels"],
            taxonomy,
        )
        repaired = repair_edited_room_polygons(
            after_plan,
            region,
            footprint,
        )
        repaired_plan = repaired["plan"]
        changed = np.asarray(before["prediction"]) != np.asarray(after["prediction"])
        baseline_preservation = vector_plan_preservation_metrics(
            before_plan,
            after_plan,
            region,
            taxonomy,
            changed_mask=changed,
        )
        repaired_preservation = vector_plan_preservation_metrics(
            before_plan,
            repaired_plan,
            region,
            taxonomy,
            changed_mask=changed,
        )
        baseline_validity = validity_checks(
            _wall_plan(after_plan),
            footprint,
            taxonomy,
            tolerances=tolerances,
            room_labels=after["room_labels"],
        )
        repaired_validity = validity_checks(
            _wall_plan(repaired_plan),
            footprint,
            taxonomy,
            tolerances=tolerances,
            room_labels=after["room_labels"],
        )
        target_graph = build_direct_room_graph(
            after["target"],
            taxonomy,
            footprint=footprint,
            tile_footprint=True,
        )
        baseline_graph = match_room_graphs(
            _graph(after_plan, after["room_labels"], taxonomy),
            target_graph,
            protocol="type_geometry_hungarian",
        )
        repaired_graph = match_room_graphs(
            _graph(repaired_plan, after["room_labels"], taxonomy),
            target_graph,
            protocol="type_geometry_hungarian",
        )
        baseline_invalid_affected = _invalid_affected_rooms(
            after_plan,
            region,
        )
        repaired_invalid_affected = _invalid_affected_rooms(
            repaired_plan,
            region,
        )
        rows.append(
            {
                "case_id": f"seed{args.seed}/{operation}/{record['name']}",
                "operation": operation,
                "repaired_rooms": repaired["repaired_rooms"],
                "overlap_adjustments": repaired["overlap_adjustments"],
                "baseline_invalid_affected_rooms": baseline_invalid_affected,
                "repaired_invalid_affected_rooms": repaired_invalid_affected,
                "baseline_preservation": baseline_preservation,
                "repaired_preservation": repaired_preservation,
                "baseline_validity": baseline_validity,
                "repaired_validity": repaired_validity,
                "baseline_graph": baseline_graph,
                "repaired_graph": repaired_graph,
            }
        )
    summary = {
        "cases": len(rows),
        "repaired_rooms": sum(row["repaired_rooms"] for row in rows),
        "overlap_adjustments": sum(
            row["overlap_adjustments"] for row in rows
        ),
        "baseline_validity": float(
            np.mean([row["baseline_validity"]["validity"] for row in rows])
        ),
        "repaired_validity": float(
            np.mean([row["repaired_validity"]["validity"] for row in rows])
        ),
        "baseline_node_f1": float(
            np.mean([row["baseline_graph"]["node_f1"] for row in rows])
        ),
        "repaired_node_f1": float(
            np.mean([row["repaired_graph"]["node_f1"] for row in rows])
        ),
        "baseline_edge_overlap": float(
            np.mean([row["baseline_graph"]["edge_overlap"] for row in rows])
        ),
        "repaired_edge_overlap": float(
            np.mean([row["repaired_graph"]["edge_overlap"] for row in rows])
        ),
        "outside_room_exact_cases": sum(
            row["repaired_preservation"]["outside_room_level_exact"] == 1.0
            for row in rows
        ),
        "baseline_invalid_affected_rooms": sum(
            row["baseline_invalid_affected_rooms"] for row in rows
        ),
        "repaired_invalid_affected_rooms": sum(
            row["repaired_invalid_affected_rooms"] for row in rows
        ),
    }
    checks = {
        "outside_room_exact_all_cases": (
            summary["outside_room_exact_cases"] == len(rows)
        ),
        "validity_not_lower": summary["repaired_validity"]
        >= summary["baseline_validity"] - 1.0e-12,
        "affected_room_invalidity_not_higher": (
            summary["repaired_invalid_affected_rooms"]
            <= summary["baseline_invalid_affected_rooms"]
        ),
        "node_f1_not_lower": summary["repaired_node_f1"]
        >= summary["baseline_node_f1"] - 1.0e-12,
        "edge_overlap_not_lower": summary["repaired_edge_overlap"]
        >= summary["baseline_edge_overlap"] - 1.0e-12,
    }
    report = {
        "status": "complete" if all(checks.values()) else "needs_iteration",
        "seed": args.seed,
        "summary": summary,
        "checks": checks,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "checks": checks}, indent=2))
    raise SystemExit(0 if report["status"] == "complete" else 1)


__all__ = ["main"]


if __name__ == "__main__":
    main()
