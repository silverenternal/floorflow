"""Canonical floorplan output and evaluator for matched SOTA experiments."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np
import torch
from shapely.geometry import Polygon

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.canonical_floorplan import (
    SCHEMA_VERSION,
    TRACKS,
    CanonicalFloorplan,
    CanonicalFurniture,
    CanonicalJunction,
    CanonicalOpening,
    CanonicalOperation,
    CanonicalRoom,
    CanonicalWall,
    canonical_fingerprint,
    validate_floorplan,
)
from floorflow.evaluation.fevp_metric_contracts import (
    FEVPTolerances,
    validity_checks,
    watertight_diagnostics,
)
from floorflow.evaluation.housemind_structure import (
    RoomTypeGraph,
    edge_set_overlap,
    housemind_style_ged,
    node_set_f1,
)
from floorflow.evaluation.room_graph_matching import (
    RoomGraph,
    RoomGraphNode,
    exact_room_graph_edit_distance,
    match_room_graphs,
)
from floorflow.evaluation.room_metric_suite import room_metric_suite
from floorflow.evaluation.wall_metrics import wall_fields
from floorflow.geometry.opening_readout import OpeningDetection, opening_detection_metrics
from floorflow.geometry.wall_graph_readout import (
    RasterWallPlan,
    ReadoutFurniture,
    ReadoutOpening,
    ReadoutWall,
)
from floorflow.representation.vector_floorplan_raster import (
    render_floorplan_rasters,
)

GRAPH_PROTOCOL = "type_geometry_hungarian"
TYPE_GRAPH_PROTOCOL = "unique_room_type_adjacency"


def _readout_plan(plan: CanonicalFloorplan) -> RasterWallPlan:
    wall_index = {wall.id: index for index, wall in enumerate(plan.walls)}
    return RasterWallPlan(
        frame=float(max(plan.width, plan.height)),
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
                "polygon": np.asarray(room.polygon, dtype=float).tolist(),
                "holes": [
                    np.asarray(hole, dtype=float).tolist()
                    for hole in room.holes
                ],
            }
            for room in plan.rooms
        ],
        openings=[
            ReadoutOpening(
                center=opening.center,
                width=opening.width,
                kind=opening.family,
                wall_index=wall_index[opening.wall_id],
            )
            for opening in plan.openings
        ],
        furniture=[
            ReadoutFurniture(
                label=item.label,
                polygon=item.polygon,
                room_index=next(
                    index
                    for index, room in enumerate(plan.rooms)
                    if room.id == item.room_id
                ),
                confidence=item.confidence,
            )
            for item in plan.furniture
        ],
    )


def _room_raster(
    plan: CanonicalFloorplan,
    taxonomy: FloorplanTaxonomy,
) -> np.ndarray:
    return render_floorplan_rasters(
        plan,
        taxonomy,
        strict=False,
    )["room_labels"]


def _polygon_mask(
    polygon: Sequence[Sequence[float]],
    shape: tuple[int, int],
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.rint(np.asarray(polygon)).astype(np.int32)], 1)
    return mask.astype(bool)


def _wall_raster(
    plan: CanonicalFloorplan,
    taxonomy: FloorplanTaxonomy,
) -> np.ndarray:
    labels = np.zeros((plan.height, plan.width), dtype=np.uint8)
    wall_id = taxonomy.class_id("wall")
    for wall in plan.walls:
        cv2.line(
            labels,
            tuple(np.rint(wall.start).astype(int)),
            tuple(np.rint(wall.end).astype(int)),
            wall_id,
            thickness=max(1, round(wall.thickness)),
        )
    return labels


def _furniture_mask(
    plan: CanonicalFloorplan,
    shape: tuple[int, int],
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for item in plan.furniture:
        mask |= _polygon_mask(item.polygon, shape)
    return mask


def _instance_graph(plan: CanonicalFloorplan) -> RoomGraph:
    index = {room.id: position for position, room in enumerate(plan.rooms)}
    nodes = tuple(
        RoomGraphNode(
            index=position,
            label=room.label,
            polygon=Polygon(room.polygon),
            centroid=(
                float(Polygon(room.polygon).centroid.x),
                float(Polygon(room.polygon).centroid.y),
            ),
            label_purity=1.0,
            second_label_fraction=0.0,
        )
        for position, room in enumerate(plan.rooms)
    )
    return RoomGraph(
        nodes=nodes,
        edges=frozenset(
            tuple(sorted((index[left], index[right])))
            for left, right in plan.adjacency
        ),
        opening_count=len(plan.openings),
    )


def _type_graph(plan: CanonicalFloorplan) -> RoomTypeGraph:
    labels = {room.id: room.label for room in plan.rooms}
    edges = {
        tuple(sorted((labels[left], labels[right])))
        for left, right in plan.adjacency
        if labels[left] != labels[right]
    }
    return RoomTypeGraph(
        labels=frozenset(room.label for room in plan.rooms),
        edges=frozenset(edges),
    )


def evaluate_canonical_pair(
    prediction: CanonicalFloorplan,
    target: CanonicalFloorplan,
    taxonomy: FloorplanTaxonomy,
    *,
    tolerances: FEVPTolerances,
    ged_timeout_seconds: float = 5.0,
) -> dict[str, Any]:
    """Evaluate one pair under fixed graph, geometry, and legality rules."""

    errors = validate_floorplan(
        prediction,
        taxonomy,
        tolerances=tolerances,
        require_complete_cad=prediction.track in {
            "rplan_generation",
            "cubicasa_reconstruction",
        },
    )
    errors.extend(validate_floorplan(target, taxonomy, tolerances=tolerances))
    if prediction.case_id != target.case_id:
        errors.append("prediction and target case ids differ")
    if prediction.track != target.track:
        errors.append("prediction and target tracks differ")
    if errors:
        raise ValueError("\n".join(errors))

    shape = (prediction.height, prediction.width)
    target_shape = (target.height, target.width)
    if shape != target_shape:
        raise ValueError("prediction and target resolutions differ")
    footprint = _polygon_mask(prediction.footprint, shape)
    target_footprint = _polygon_mask(target.footprint, target_shape)
    room_labels = _room_raster(prediction, taxonomy)
    target_labels = _room_raster(target, taxonomy)
    room_suite = room_metric_suite(
        room_labels,
        target_labels,
        footprint,
        taxonomy,
        predicted_room_support=footprint,
        target_room_support=target_footprint,
    )
    wall = wall_fields(
        torch.from_numpy(_wall_raster(prediction, taxonomy))[None],
        torch.from_numpy(_wall_raster(target, taxonomy))[None],
        taxonomy,
    )
    predicted_openings = [
        OpeningDetection(
            center=opening.center,
            width=opening.width,
            family=opening.family,
            class_id=taxonomy.class_id(opening.label),
            score=opening.confidence,
            wall_index=None,
            wall_distance=None,
        )
        for opening in prediction.openings
    ]
    target_openings = [
        OpeningDetection(
            center=opening.center,
            width=opening.width,
            family=opening.family,
            class_id=taxonomy.class_id(opening.label),
            score=opening.confidence,
            wall_index=None,
            wall_distance=None,
        )
        for opening in target.openings
    ]
    opening = opening_detection_metrics(
        predicted_openings,
        target_openings,
        tolerance=1.0,
    )
    predicted_furniture = _furniture_mask(prediction, shape)
    target_furniture = _furniture_mask(target, target_shape)
    furniture_union = np.logical_or(
        predicted_furniture,
        target_furniture,
    ).sum()
    furniture_f1 = (
        float(
            2.0
            * np.logical_and(predicted_furniture, target_furniture).sum()
            / max(
                1,
                predicted_furniture.sum() + target_furniture.sum(),
            )
        )
        if furniture_union
        else 1.0
    )
    predicted_plan = _readout_plan(prediction)
    validity = validity_checks(
        predicted_plan,
        footprint,
        taxonomy,
        tolerances=tolerances,
        room_labels=room_labels,
    )
    watertight = watertight_diagnostics(
        predicted_plan,
        taxonomy=taxonomy,
        tolerances=tolerances,
        labels=room_labels,
        footprint=footprint,
    )
    instance = match_room_graphs(
        _instance_graph(prediction),
        _instance_graph(target),
        protocol=GRAPH_PROTOCOL,
    )
    instance_ged = exact_room_graph_edit_distance(
        _instance_graph(prediction),
        _instance_graph(target),
        timeout_seconds=ged_timeout_seconds,
    )
    predicted_type = _type_graph(prediction)
    target_type = _type_graph(target)
    type_nodes = node_set_f1(predicted_type, target_type)
    type_edges = edge_set_overlap(predicted_type, target_type)
    type_ged = housemind_style_ged(
        predicted_type,
        target_type,
        timeout_seconds=ged_timeout_seconds,
    )
    matched = room_suite["matched_instance"]
    return {
        "case_id": prediction.case_id,
        "track": prediction.track,
        "graph_protocol": GRAPH_PROTOCOL,
        "type_graph_protocol": TYPE_GRAPH_PROTOCOL,
        "room_iou": float(matched["matched_geometric_iou"]),
        "instance_f1": float(matched["instance_f1_at_threshold"]),
        "wall_f1": float(wall["wall_centerline_f1"]),
        "wall_iou": float(wall["wall_iou"]),
        "opening_f1": float(opening["f1"]),
        "furniture_f1": furniture_f1,
        "node_precision": float(type_nodes["precision"]),
        "node_recall": float(type_nodes["recall"]),
        "node_f1": float(type_nodes["f1"]),
        "edge_precision": float(type_edges["precision"]),
        "edge_recall": float(type_edges["recall"]),
        "edge_overlap": float(type_edges["overlap"]),
        "ged": float(type_ged["ged"]),
        "ged_exact": bool(type_ged["exact"]),
        "instance_node_f1": float(instance["node_f1"]),
        "instance_edge_overlap": float(instance["edge_overlap"]),
        "instance_ged": float(instance_ged["ged"]),
        "instance_ged_exact": bool(instance_ged["ged_exact"]),
        "validity": float(validity["validity"]),
        "conditional_watertightness": float(watertight["watertight_rate"]),
        "hole_count": int(watertight["hole_count"]),
        "hole_area_px": float(watertight["hole_area_px"]),
        "disconnected_component_count": int(
            watertight["disconnected_component_count"]
        ),
        "dangling_endpoint_rate": float(
            watertight["dangling_endpoint_rate"]
        ),
        "watertight_issue_codes": tuple(watertight["watertight_issue_codes"]),
    }


__all__ = [
    "GRAPH_PROTOCOL",
    "SCHEMA_VERSION",
    "TRACKS",
    "TYPE_GRAPH_PROTOCOL",
    "CanonicalFloorplan",
    "CanonicalFurniture",
    "CanonicalJunction",
    "CanonicalOpening",
    "CanonicalOperation",
    "CanonicalRoom",
    "CanonicalWall",
    "canonical_fingerprint",
    "evaluate_canonical_pair",
    "validate_floorplan",
]
