"""Raster2Seq-style room corner and angle metrics over semantic label maps."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.fevp_metric_contracts import (
    RoomInstance,
)
from floorflow.geometry.partition_polygons import explicit_partition_plan
from floorflow.geometry.wall_graph_readout import raster_labels_to_wall_plan


def _plan_rooms(
    labels: np.ndarray,
    taxonomy: FloorplanTaxonomy,
    minimum_area_px: float,
    *,
    footprint: np.ndarray | None = None,
    explicit_rooms: bool = False,
) -> list[RoomInstance]:
    plan = (
        explicit_partition_plan(labels, footprint, taxonomy)
        if explicit_rooms and footprint is not None
        else raster_labels_to_wall_plan(labels, taxonomy)
    )
    rooms = []
    for row in plan.rooms:
        polygon = Polygon(
            row["polygon"],
            row.get("holes", []),
        )
        if polygon.is_empty or polygon.area < float(minimum_area_px):
            continue
        rooms.append(
            RoomInstance(
                index=len(rooms),
                label=str(row["label"]),
                mask=np.zeros(labels.shape, dtype=bool),
                polygon=polygon,
                area=float(polygon.area),
            )
        )
    return rooms


def _vertices(instance: RoomInstance) -> np.ndarray:
    polygon = orient(instance.polygon, sign=1.0)
    return np.asarray(polygon.exterior.coords[:-1], dtype=np.float64)


def _room_iou(left: RoomInstance, right: RoomInstance) -> float:
    union = left.polygon.union(right.polygon).area
    if union <= 0.0:
        return 0.0
    return float(left.polygon.intersection(right.polygon).area / union)


def _matched_rooms(
    predicted: list[RoomInstance],
    target: list[RoomInstance],
    minimum_iou: float,
) -> list[tuple[RoomInstance, RoomInstance]]:
    if not predicted or not target:
        return []
    scores = np.zeros((len(predicted), len(target)), dtype=np.float64)
    for left, prediction in enumerate(predicted):
        for right, expected in enumerate(target):
            scores[left, right] = _room_iou(prediction, expected)
    rows, columns = linear_sum_assignment(-scores)
    return [
        (predicted[row], target[column])
        for row, column in zip(rows, columns, strict=True)
        if scores[row, column] >= float(minimum_iou)
    ]


def _turn_angles(vertices: np.ndarray) -> np.ndarray:
    previous = np.roll(vertices, 1, axis=0)
    following = np.roll(vertices, -1, axis=0)
    incoming = previous - vertices
    outgoing = following - vertices
    cross = incoming[:, 0] * outgoing[:, 1] - incoming[:, 1] * outgoing[:, 0]
    dot = (incoming * outgoing).sum(axis=1)
    return np.degrees(np.arctan2(cross, dot))


def _corner_matches(
    predicted: np.ndarray,
    target: np.ndarray,
    tolerance_px: float,
) -> list[tuple[int, int, float]]:
    matches = []
    for target_index, point in enumerate(target):
        distances = np.linalg.norm(predicted - point, axis=1)
        predicted_index = int(np.argmin(distances))
        distance = float(distances[predicted_index])
        if distance <= float(tolerance_px):
            matches.append((predicted_index, target_index, distance))
    return matches


def room_corner_angle_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    taxonomy: FloorplanTaxonomy,
    *,
    minimum_room_iou: float = 0.5,
    corner_tolerance_px: float = 10.0,
    angle_tolerance_degrees: float = 5.0,
    minimum_area_px: float = 4.0,
    footprint: np.ndarray | None = None,
    explicit_rooms: bool = False,
) -> dict[str, float]:
    """Apply Raster2Seq's frozen corner/angle tolerances to matched rooms."""

    predicted_rooms = _plan_rooms(
        np.asarray(prediction),
        taxonomy,
        minimum_area_px,
        footprint=footprint,
        explicit_rooms=explicit_rooms,
    )
    target_rooms = _plan_rooms(
        np.asarray(target),
        taxonomy,
        minimum_area_px,
        footprint=footprint,
        explicit_rooms=explicit_rooms,
    )
    matches = _matched_rooms(
        predicted_rooms,
        target_rooms,
        minimum_room_iou,
    )
    predicted_matched: set[int] = set()
    target_matched: set[int] = set()
    angle_matched = 0
    angle_total = 0
    orthogonal = 0
    predicted_corners = 0
    for predicted_room, target_room in matches:
        predicted_vertices = _vertices(predicted_room)
        target_vertices = _vertices(target_room)
        predicted_corners += len(predicted_vertices)
        for predicted_index, target_index, _distance in _corner_matches(
            predicted_vertices,
            target_vertices,
            corner_tolerance_px,
        ):
            predicted_matched.add(id(predicted_room) * 100_003 + predicted_index)
            target_matched.add(id(target_room) * 100_003 + target_index)
            angle_total += 1
            predicted_angle = _turn_angles(predicted_vertices)[predicted_index]
            target_angle = _turn_angles(target_vertices)[target_index]
            if abs(predicted_angle - target_angle) <= float(
                angle_tolerance_degrees
            ):
                angle_matched += 1
        for angle in _turn_angles(predicted_vertices):
            folded = abs(angle) % 180.0
            angle_error = min(
                folded,
                abs(90.0 - folded),
                abs(180.0 - folded),
            )
            if angle_error <= float(angle_tolerance_degrees):
                orthogonal += 1
    target_corners = sum(len(_vertices(room)) for room in target_rooms)
    precision = len(predicted_matched) / max(1, predicted_corners)
    recall = len(target_matched) / max(1, target_corners)
    corner_f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0.0
        else 0.0
    )
    return {
        "corner_precision": float(precision),
        "corner_recall": float(recall),
        "corner_f1": float(corner_f1),
        "angle_f1": float(angle_matched / max(1, angle_total)),
        "orthogonal_turn_rate": float(
            orthogonal / max(1, predicted_corners)
        ),
        "matched_rooms_for_corners": float(len(matches)),
    }


__all__ = ["room_corner_angle_metrics"]
