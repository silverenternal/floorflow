"""Adapters for fpvec-lab's MIT-licensed fpeval implementation."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.geometry.opening_readout import read_wall_interval_detections
from floorflow.geometry.reconciliation import reconcile_plan
from floorflow.geometry.room_polygons import inferred_footprint, room_polygon_rows
from floorflow.geometry.wall_graph_readout import (
    RasterWallPlan,
    ReadoutOpening,
    raster_labels_to_wall_plan,
)

FPVEC_ROOT = Path(__file__).resolve().parents[3] / "third_party/fpvec-lab"
if str(FPVEC_ROOT) not in sys.path:
    sys.path.insert(0, str(FPVEC_ROOT))

from fpeval.metrics import evaluate
from fpeval.model import from_vlm_json
from fpeval.verify import verify


def to_fpeval_plan(plan: RasterWallPlan):
    return from_vlm_json(plan.to_json(), frame=plan.frame)


def summarise_fpeval_report(report) -> dict[str, float | int | None]:
    """Flatten the fields consumed by the project publication gate."""

    return {
        "validity": float(report.r_val),
        "ext_iou": float(report.ext_iou),
        "wall_f1": float(report.walls["f1"]),
        "wall_precision": float(report.walls["precision"]),
        "wall_recall": float(report.walls["recall"]),
        "wall_mean_distance": report.walls["mean_dist"],
        "opening_f1": float(report.openings["f1"]),
        "opening_precision": float(report.openings["precision"]),
        "opening_recall": float(report.openings["recall"]),
        "room_f1": float(report.rooms["f1"]),
        "room_precision": float(report.rooms["precision"]),
        "room_recall": float(report.rooms["recall"]),
        "room_mean_iou": report.rooms["mean_iou"],
        "room_label_accuracy": report.rooms["label_acc"],
        "edit_cost": float(report.edit["total"]),
        "edit_cost_per_gt_wall": float(report.edit["per_gt_wall"]),
        "verify_issue_count": len(report.verify_issues),
    }


def count_double_walls(plan, tolerance_fraction: float = 0.003) -> int:
    """Count near-coincident predicted centerlines without direction aliasing."""

    tolerance = max(1.0e-6, float(tolerance_fraction) * plan.frame)
    count = 0
    for index, left in enumerate(plan.walls):
        left_vector = (
            left.end[0] - left.start[0],
            left.end[1] - left.start[1],
        )
        left_length = math.hypot(*left_vector)
        for right in plan.walls[index + 1 :]:
            right_vector = (
                right.end[0] - right.start[0],
                right.end[1] - right.start[1],
            )
            right_length = math.hypot(*right_vector)
            if min(left_length, right_length) <= tolerance:
                continue
            cosine = abs(
                left_vector[0] * right_vector[0]
                + left_vector[1] * right_vector[1]
            ) / (left_length * right_length)
            if cosine < math.cos(math.radians(5.0)):
                continue
            direct = max(
                math.dist(left.start, right.start),
                math.dist(left.end, right.end),
            )
            reversed_pair = max(
                math.dist(left.start, right.end),
                math.dist(left.end, right.start),
            )
            if min(direct, reversed_pair) <= tolerance:
                count += 1
    return count


def _attach_interval_openings(
    plan: RasterWallPlan,
    foreground_probability: np.ndarray,
    opening_logits: np.ndarray,
    taxonomy: FloorplanTaxonomy,
    *,
    threshold: float,
) -> int:
    """Replace label-derived openings by wall-interval heatmap readout."""

    if foreground_probability.ndim != 2 or opening_logits.ndim != 3:
        raise ValueError(
            "opening heatmap readout expects [H,W] foreground and [K,H,W] logits"
        )
    if opening_logits.shape[0] <= 0:
        return 0
    opening_ids = tuple(taxonomy.opening_ids[: opening_logits.shape[0]])
    detections = read_wall_interval_detections(
        foreground_probability,
        opening_logits,
        opening_ids,
        taxonomy.class_names,
        plan,
        threshold=float(threshold),
    )
    if not detections:
        return 0
    existing = list(plan.openings)
    dedupe = max(2.0, 0.02 * float(plan.frame))
    added = []
    for detection in detections:
        if any(
            math.dist(detection.center, opening.center) <= dedupe
            for opening in existing
        ):
            continue
        added.append(
            ReadoutOpening(
                center=detection.center,
                width=detection.width,
                kind=detection.family,
                wall_index=int(detection.wall_index or 0),
            )
        )
        existing.append(added[-1])
    if not added:
        return 0
    if plan.openings and len(added) > max(2, len(plan.openings)):
        # A heatmap that adds far more openings than the label set is not
        # calibrated enough to trust; keep the label-derived openings.
        return 0
    plan.openings = existing
    return len(added)


def attach_wall_interval_openings(
    plan: RasterWallPlan,
    foreground_probability: np.ndarray,
    opening_logits: np.ndarray,
    taxonomy: FloorplanTaxonomy,
    *,
    threshold: float = 0.5,
) -> int:
    """Public adapter for host-wall opening interval readout."""

    return _attach_interval_openings(
        plan,
        foreground_probability,
        opening_logits,
        taxonomy,
        threshold=threshold,
    )


def fpeval_raster_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    taxonomy: FloorplanTaxonomy,
    *,
    tolerance_fractions: tuple[float, ...] = (0.015, 0.05),
    explicit_prediction_rooms: bool = False,
    explicit_target_rooms: bool = False,
    tile_explicit_rooms: bool = False,
    target_plan=None,
    reconcile: bool = False,
    reconcile_use_ip: bool = False,
    prediction_opening_probability: np.ndarray | None = None,
    prediction_opening_logits: np.ndarray | None = None,
    prediction_opening_threshold: float = 0.5,
) -> dict:
    pred_plan = raster_labels_to_wall_plan(prediction, taxonomy)
    if reconcile:
        pred_plan = reconcile_plan(
            pred_plan,
            prediction,
            taxonomy,
            use_ip=reconcile_use_ip,
        )
    opening_source = "labels"
    if prediction_opening_probability is not None or prediction_opening_logits is not None:
        if prediction_opening_probability is None or prediction_opening_logits is None:
            raise ValueError(
                "opening heatmap readout needs both probability and logits"
            )
        read_count = _attach_interval_openings(
            pred_plan,
            prediction_opening_probability,
            prediction_opening_logits,
            taxonomy,
            threshold=prediction_opening_threshold,
        )
        opening_source = "heatmap_fusion" if read_count else "labels"
    pred_plan = to_fpeval_plan(pred_plan)
    if explicit_prediction_rooms:
        prediction_doc = floorplan_to_json(pred_plan)
        footprint = (
            inferred_footprint(prediction)
            if tile_explicit_rooms
            else (
                np.isin(prediction, np.asarray(taxonomy.room_ids))
                | np.isin(prediction, np.asarray(taxonomy.opening_ids))
                | (prediction == taxonomy.class_id("wall"))
            )
        )
        prediction_doc["rooms"] = room_polygon_rows(
            prediction,
            taxonomy,
            footprint=footprint,
            tile_footprint=tile_explicit_rooms,
        )
        pred_plan = from_vlm_json(prediction_doc, frame=pred_plan.frame)
    if target_plan is None:
        target_plan = to_fpeval_plan(
            raster_labels_to_wall_plan(target, taxonomy)
        )
    if explicit_target_rooms:
        target_doc = floorplan_to_json(target_plan)
        footprint = (
            inferred_footprint(target)
            if tile_explicit_rooms
            else (
                np.isin(target, np.asarray(taxonomy.room_ids))
                | np.isin(target, np.asarray(taxonomy.opening_ids))
                | (target == taxonomy.class_id("wall"))
            )
        )
        target_doc["rooms"] = room_polygon_rows(
            target,
            taxonomy,
            footprint=footprint,
            tile_footprint=tile_explicit_rooms,
        )
        target_plan = from_vlm_json(target_doc, frame=target_plan.frame)
    reports = {}
    summaries = {}
    for tolerance in tolerance_fractions:
        report = evaluate(pred_plan, target_plan, tol_frac=float(tolerance))
        reports[str(tolerance)] = report.as_dict()
        summaries[str(tolerance)] = summarise_fpeval_report(report)
    pred_validity = verify(pred_plan).as_dict()
    target_validity = verify(target_plan).as_dict()
    return {
        "prediction": floorplan_to_json(pred_plan),
        "target": floorplan_to_json(target_plan),
        "tolerances": reports,
        "summary": summaries,
        "prediction_validity": pred_validity,
        "target_validity": target_validity,
        "prediction_double_walls": count_double_walls(pred_plan),
        "opening_readout_source": opening_source,
        "reconciled": bool(reconcile),
    }


def prepare_target_plan(
    target: np.ndarray,
    taxonomy: FloorplanTaxonomy,
):
    """Prepare the reusable target plan once for multiple predictions."""

    return to_fpeval_plan(raster_labels_to_wall_plan(target, taxonomy))


def floorplan_to_json(plan) -> dict:
    return {
        "walls": [
            {
                "id": wall.id,
                "start": list(wall.start),
                "end": list(wall.end),
                "thickness": wall.thickness,
                "type": wall.type,
                "openings": [
                    {
                        "type": opening.kind,
                        "center": list(opening.center),
                        "width": opening.width,
                    }
                    for opening in wall.openings
                ],
            }
            for wall in plan.walls
        ],
        "rooms": [
            {
                "label": room.label,
                "walls": list(room.wall_ids),
                "polygon": room.polygon,
                "holes": list(getattr(room, "holes", [])),
            }
            for room in plan.rooms
        ],
        "frame": plan.frame,
    }
