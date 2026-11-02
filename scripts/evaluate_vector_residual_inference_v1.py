#!/usr/bin/env python3
"""Evaluate the trained vector residual refiner on held-out FloorFlow edits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from floorflow.data.geometry_cache import geometry_from_payload, load_geometry_cache, unpack_polygons
from floorflow.evaluation.typed_edit_metrics import normalized_region_mask
from floorflow.evaluation.vector_edit_protocol import (
    extract_vector_plan,
    vector_plan_preservation_metrics,
)
from floorflow.heads.vector_proposal_refiner import VectorProposalRefiner
from floorflow.inference.vector_residual_refiner import refine_vector_plan
from floorflow.training.checkpoint import load_checkpoint_model

ROOT = Path(__file__).resolve().parents[1]
EDITS = {
    "room_type": "fp_local_edit_room_type_test503_v1",
    "room_add": "fp_local_edit_room_add_test503_v1",
    "room_remove": "fp_local_edit_room_remove_test503_v1",
    "opening_delete": "fp_local_edit_opening_delete_test503_v1",
    "furniture_delete": "fp_local_edit_furniture_delete_test503_v1",
}


def _sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backbone",
        type=Path,
        default=ROOT / "outputs/train/condition_cfm_1024_3ep_round2/best.pt",
    )
    parser.add_argument(
        "--refiner",
        type=Path,
        default=ROOT / "outputs/train/vector_proposal_refiner_smoke_v1/best.pt",
    )
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--limit-per-operation", type=int, default=100)
    parser.add_argument("--refine-rooms", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/vector_residual_inference_v1.json",
    )
    return parser.parse_args()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as payload:
        return {key: payload[key] for key in payload.files}


def _resample(points: np.ndarray, count: int = 16) -> np.ndarray:
    values = np.asarray(points, dtype=np.float32)
    closed = np.concatenate([values, values[:1]], axis=0)
    lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    samples = np.linspace(0.0, cumulative[-1], count, endpoint=False)
    result = np.empty((count, 2), dtype=np.float32)
    result[:, 0] = np.interp(samples, cumulative, closed[:, 0])
    result[:, 1] = np.interp(samples, cumulative, closed[:, 1])
    return result


def _chamfer(left: np.ndarray, right: np.ndarray) -> float:
    distances = np.linalg.norm(left[:, None] - right[None], axis=-1)
    return float(
        0.5
        * (
            distances.min(axis=1).mean()
            + distances.min(axis=0).mean()
        )
    )


def _room_error(plan, target_polygons, target_labels, affected, scale):
    if not plan.rooms or not target_polygons:
        return None
    pred_centers = np.asarray(
        [
            [room.polygon.centroid.x / scale, room.polygon.centroid.y / scale]
            for room in plan.rooms
        ]
    )
    target_centers = np.asarray(
        [polygon.mean(axis=0) / scale for polygon in target_polygons]
    )
    cost = np.linalg.norm(pred_centers[:, None] - target_centers[None], axis=-1)
    for left, room in enumerate(plan.rooms):
        for right, label in enumerate(target_labels):
            if room.label != label:
                cost[left, right] += 0.5
    rows, columns = linear_sum_assignment(cost)
    errors = [
        _chamfer(
            _resample(np.asarray(plan.rooms[row].polygon.exterior.coords)[:-1] / scale),
            _resample(target_polygons[column] / scale),
        )
        for row, column in zip(rows, columns, strict=True)
        if row < len(affected) and affected[row]
    ]
    return float(np.mean(errors)) if errors else None


def _wall_error(plan, target_lines, affected, scale):
    if not plan.walls or not target_lines.size:
        return None
    pred_centers = np.asarray(
        [
            [
                (wall.start[0] + wall.end[0]) / (2.0 * scale),
                (wall.start[1] + wall.end[1]) / (2.0 * scale),
            ]
            for wall in plan.walls
        ]
    )
    target_centers = target_lines.mean(axis=1) / scale
    cost = np.linalg.norm(pred_centers[:, None] - target_centers[None], axis=-1)
    rows, columns = linear_sum_assignment(cost)
    errors = []
    for row, column in zip(rows, columns, strict=True):
        if row >= len(affected) or not affected[row]:
            continue
        wall = np.asarray(
            [plan.walls[row].start, plan.walls[row].end],
            dtype=np.float32,
        ) / scale
        target = target_lines[column] / scale
        direct = np.abs(wall - target).mean()
        reverse = np.abs(wall - target[::-1]).mean()
        errors.append(float(min(direct, reverse)))
    return float(np.mean(errors)) if errors else None


def _opening_error(plan, target_centers, affected, scale):
    if not plan.openings or not target_centers.size:
        return None
    pred = np.asarray([opening.center for opening in plan.openings]) / scale
    target = target_centers / scale
    cost = np.linalg.norm(pred[:, None] - target[None], axis=-1)
    rows, columns = linear_sum_assignment(cost)
    errors = [
        float(np.linalg.norm(pred[row] - target[column]))
        for row, column in zip(rows, columns, strict=True)
        if row < len(affected) and affected[row]
    ]
    return float(np.mean(errors)) if errors else None


def _features(backbone, footprint, floor_id):
    target = torch.zeros(
        1,
        backbone.target_channels,
        *footprint.shape[-2:],
        device=footprint.device,
    )
    with torch.no_grad():
        return backbone._encode_features(
            target,
            torch.zeros(1, device=footprint.device),
            footprint[None].float(),
            floor_ids=torch.tensor([floor_id], device=footprint.device),
        )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    backbone, _backlog_config, taxonomy, _ = load_checkpoint_model(args.backbone, device)
    backbone.requires_grad_(False)
    refiner_config = torch.load(args.refiner, map_location="cpu", weights_only=False)
    refiner = VectorProposalRefiner(
        in_channels=backbone.output.in_channels,
        hidden_dim=int(refiner_config["config"]["hidden_dim"]),
    ).to(device)
    refiner.load_state_dict(refiner_config["refiner"])
    refiner.eval()
    rows = []
    for operation, directory in EDITS.items():
        manifest = json.loads(
            (
                ROOT / "outputs/control_contracts" / directory / "test/manifest.json"
            ).read_text(encoding="utf-8")
        )
        records = [row for row in manifest["rows"] if row["edited"]]
        for record in records[: int(args.limit_per_operation)]:
            safe_name = str(record["name"]).replace("/", "__")
            before = _load_npz(
                ROOT
                / f"outputs/eval/fp_hard_metrics_full_test503_seed{args.seed}_v1/rasters"
                / f"{safe_name}.npz"
            )
            after = _load_npz(
                ROOT
                / f"outputs/eval/fp_control_gated_local_test503_seed{args.seed}_v1"
                / operation
                / f"{safe_name}.npz"
            )
            footprint = torch.from_numpy(after["footprint"].astype(np.float32)).to(device)
            region = normalized_region_mask(
                after["footprint"].shape,
                record["edit_region_normalized"],
            ) & after["footprint"].astype(bool)
            plan = extract_vector_plan(
                after["prediction"],
                after["room_labels"],
                taxonomy,
            )
            style, rest = str(record["name"]).split("/", 1)
            case_id, floor = rest.split("__", 1)
            geometry_path = (
                ROOT
                / "outputs/cache/cubicasa5k_geometry_v1/test"
                / style
                / case_id
                / f"{floor.replace('-', '_')}.geometry.npz"
            )
            geometry = geometry_from_payload(load_geometry_cache(geometry_path))
            target_rooms = unpack_polygons(geometry.rooms)
            target_labels = [
                taxonomy.class_names[int(class_id)]
                for class_id in geometry.room_class_ids
            ]
            floor_id = int(floor.rsplit("-", 1)[-1])
            features = _features(backbone, footprint, floor_id)
            refined = refine_vector_plan(
                plan,
                region,
                features,
                refiner,
                refine_rooms=args.refine_rooms,
            )
            before_plan = extract_vector_plan(
                before["prediction"],
                before["room_labels"],
                taxonomy,
            )
            changed = np.asarray(before["prediction"]) != np.asarray(after["prediction"])
            baseline_preservation = vector_plan_preservation_metrics(
                before_plan,
                plan,
                region,
                taxonomy,
                changed_mask=changed,
            )
            refined_preservation = vector_plan_preservation_metrics(
                before_plan,
                refined.plan,
                region,
                taxonomy,
                changed_mask=changed,
            )
            scale = float(after["prediction"].shape[-1])
            rows.append(
                {
                    "case_id": f"seed{args.seed}/{operation}/{record['name']}",
                    "operation": operation,
                    "affected_rooms": refined.affected_rooms,
                    "affected_walls": refined.affected_walls,
                    "affected_openings": refined.affected_openings,
                    "outside_baseline": baseline_preservation["outside_room_level_exact"],
                    "outside_refined": refined_preservation["outside_room_level_exact"],
                    "preservation": {
                        "baseline": baseline_preservation,
                        "refined": refined_preservation,
                    },
                    "room_baseline": _room_error(
                        plan,
                        target_rooms,
                        target_labels,
                        refined.room_mask,
                        scale,
                    ),
                    "room_refined": _room_error(
                        refined.plan,
                        target_rooms,
                        target_labels,
                        refined.room_mask,
                        scale,
                    ),
                    "wall_baseline": _wall_error(
                        plan,
                        geometry.wall_centerlines,
                        refined.wall_mask,
                        scale,
                    ),
                    "wall_refined": _wall_error(
                        refined.plan,
                        geometry.wall_centerlines,
                        refined.wall_mask,
                        scale,
                    ),
                    "opening_baseline": _opening_error(
                        plan,
                        geometry.opening_centers,
                        refined.opening_mask,
                        scale,
                    ),
                    "opening_refined": _opening_error(
                        refined.plan,
                        geometry.opening_centers,
                        refined.opening_mask,
                        scale,
                    ),
                }
            )
    summary = {}
    for operation in EDITS:
        selected = [row for row in rows if row["operation"] == operation]
        summary[operation] = {}
        for family in ("room", "wall", "opening"):
            pairs = [
                (row[f"{family}_baseline"], row[f"{family}_refined"])
                for row in selected
                if row[f"{family}_baseline"] is not None
                and row[f"{family}_refined"] is not None
            ]
            summary[operation][family] = {
                "cases": len(pairs),
                "baseline_mae": float(np.mean([pair[0] for pair in pairs]))
                if pairs
                else None,
                "refined_mae": float(np.mean([pair[1] for pair in pairs]))
                if pairs
                else None,
            }
    checks = {
        "outside_exact_all_cases": all(row["outside_refined"] == 1.0 for row in rows),
        "outside_vector_layers_exact_all_cases": all(
            row["preservation"]["refined"][metric] == 1.0
            for row in rows
            for metric in (
                "outside_room_label_exact",
                "outside_room_polygon_exact",
                "outside_wall_exact",
                "outside_opening_exact",
                "outside_edge_exact",
            )
        ),
        "locality_violation_area_zero_all_cases": all(
            row["preservation"]["refined"]["locality_violation_area"] == 0.0
            for row in rows
        ),
        "room_not_worse": all(
            row["room_baseline"] is None
            or row["room_refined"] is None
            or row["room_refined"] <= row["room_baseline"] + 1.0e-9
            for row in rows
        ),
        "wall_improved": sum(
            (row["wall_baseline"] or 0.0) > (row["wall_refined"] or 0.0)
            for row in rows
        )
        > sum(
            (row["wall_baseline"] or 0.0) < (row["wall_refined"] or 0.0)
            for row in rows
        ),
        "opening_improved": sum(
            (row["opening_baseline"] or 0.0) > (row["opening_refined"] or 0.0)
            for row in rows
        )
        > sum(
            (row["opening_baseline"] or 0.0) < (row["opening_refined"] or 0.0)
            for row in rows
        ),
    }
    report = {
        "status": "complete" if all(checks.values()) else "needs_iteration",
        "seed": args.seed,
        "cases": len(rows),
        "limit_per_operation": int(args.limit_per_operation),
        "summary": summary,
        "checks": checks,
        "room_refinement_enabled": bool(args.refine_rooms),
        "backbone_checkpoint": str(args.backbone),
        "backbone_sha256": _sha256(args.backbone),
        "refiner_checkpoint": str(args.refiner),
        "refiner_sha256": _sha256(args.refiner),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "summary": summary, "checks": checks}, indent=2))
    raise SystemExit(0 if report["status"] == "complete" else 1)


__all__ = ["main"]


if __name__ == "__main__":
    main()
