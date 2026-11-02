"""Supplementary structural-legality metrics for vectorized floorplans."""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon
from shapely.ops import unary_union

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.geometry.wall_graph_readout import raster_labels_to_wall_plan


def _polygons(rooms: list[dict]) -> tuple[list[Polygon], list[str]]:
    polygons = []
    labels = []
    for room in rooms:
        polygon = Polygon(room["polygon"])
        if polygon.is_empty or polygon.area <= 0.0:
            continue
        polygons.append(polygon)
        labels.append(str(room["label"]))
    return polygons, labels


def _graph(polygons: list[Polygon], labels: list[str], tolerance: float) -> set:
    edges = set()
    for left in range(len(polygons)):
        for right in range(left + 1, len(polygons)):
            if polygons[left].boundary.distance(polygons[right].boundary) > tolerance:
                continue
            edges.add(tuple(sorted((labels[left], labels[right]))))
    return edges


def _iou(left: Polygon, right: Polygon) -> float:
    union = left.union(right).area
    return float(left.intersection(right).area / union) if union > 0 else 0.0


def _node_matching(
    predicted: list[Polygon],
    predicted_labels: list[str],
    target: list[Polygon],
    target_labels: list[str],
) -> dict[int, int]:
    if not predicted or not target:
        return {}
    similarity = np.zeros((len(predicted), len(target)), dtype=np.float64)
    for left, (pred_polygon, pred_label) in enumerate(
        zip(predicted, predicted_labels, strict=True)
    ):
        for right, (target_polygon, target_label) in enumerate(
            zip(target, target_labels, strict=True)
        ):
            if pred_label == target_label:
                similarity[left, right] = _iou(pred_polygon, target_polygon)
    rows, columns = linear_sum_assignment(-similarity)
    return {
        int(row): int(column)
        for row, column in zip(rows, columns, strict=True)
        if similarity[row, column] >= 0.5
    }


def wall_junction_summary(walls, *, tolerance: float = 2.0) -> dict[str, float | int]:
    if not walls:
        return {
            "wall_continuity_rate": 0.0,
            "junction_count": 0,
            "open_endpoint_count": 0,
            "degree_ge_2_rate": 0.0,
            "degree_ge_3_rate": 0.0,
        }
    endpoints = [
        point
        for wall in walls
        for point in (wall.start, wall.end)
    ]
    clusters = []
    for point in endpoints:
        for cluster in clusters:
            center = (
                sum(item[0] for item in cluster) / len(cluster),
                sum(item[1] for item in cluster) / len(cluster),
            )
            if math.dist(point, center) <= tolerance:
                cluster.append(point)
                break
        else:
            clusters.append([point])
    degrees = [len(cluster) for cluster in clusters]
    return {
        "wall_continuity_rate": float(np.mean([degree >= 2 for degree in degrees])),
        "junction_count": len(clusters),
        "open_endpoint_count": int(sum(degree < 2 for degree in degrees)),
        "degree_ge_2_rate": float(np.mean([degree >= 2 for degree in degrees])),
        "degree_ge_3_rate": float(np.mean([degree >= 3 for degree in degrees])),
    }


def structural_legality_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    taxonomy: FloorplanTaxonomy,
) -> dict[str, float]:
    pred_plan = raster_labels_to_wall_plan(prediction, taxonomy)
    target_plan = raster_labels_to_wall_plan(target, taxonomy)
    pred_polygons, pred_labels = _polygons(pred_plan.rooms)
    target_polygons, target_labels = _polygons(target_plan.rooms)
    matches = _node_matching(
        pred_polygons,
        pred_labels,
        target_polygons,
        target_labels,
    )

    overlap = 0.0
    for left in range(len(pred_polygons)):
        for right in range(left + 1, len(pred_polygons)):
            overlap += pred_polygons[left].intersection(pred_polygons[right]).area
    pred_union = unary_union(pred_polygons).area if pred_polygons else 0.0
    frame = float(max(prediction.shape))
    outside = sum(
        polygon.difference(
            Polygon([(0, 0), (frame, 0), (frame, frame), (0, frame)])
        ).area
        for polygon in pred_polygons
    )
    tolerance = 2.0
    pred_edges = _graph(pred_polygons, pred_labels, tolerance)
    target_edges = _graph(target_polygons, target_labels, tolerance)
    mapped_edges = set()
    reverse = {pred: target for pred, target in matches.items()}
    predicted_ids = list(range(len(pred_polygons)))
    for left in predicted_ids:
        for right in range(left + 1, len(predicted_ids)):
            if left not in reverse or right not in reverse:
                continue
            if (
                pred_polygons[left].boundary.distance(
                    pred_polygons[right].boundary
                )
                > tolerance
            ):
                continue
            mapped_edges.add(tuple(sorted((reverse[left], reverse[right]))))
    target_edge_indices = set()
    for left in range(len(target_polygons)):
        for right in range(left + 1, len(target_polygons)):
            if (
                target_polygons[left].boundary.distance(
                    target_polygons[right].boundary
                )
                <= tolerance
            ):
                target_edge_indices.add((left, right))
    edge_union = mapped_edges | target_edge_indices
    edge_overlap = (
        len(mapped_edges & target_edge_indices) / len(edge_union)
        if edge_union
        else 1.0
    )
    matched = len(matches)
    node_f1 = (
        2.0 * matched / (len(pred_polygons) + len(target_polygons))
        if pred_polygons or target_polygons
        else 1.0
    )
    ged = (
        len(pred_polygons)
        + len(target_polygons)
        - 2 * matched
        + len(mapped_edges.symmetric_difference(target_edge_indices))
    )
    return {
        "parsable": float(bool(pred_polygons and pred_plan.walls)),
        "room_closure_rate": float(
            np.mean([polygon.is_valid for polygon in pred_polygons])
            if pred_polygons
            else 0.0
        ),
        "wall_continuity_rate": wall_junction_summary(pred_plan.walls)[
            "wall_continuity_rate"
        ],
        "self_intersection_rate": float(
            1.0
            - np.mean([polygon.is_valid for polygon in pred_polygons])
            if pred_polygons
            else 1.0
        ),
        "room_overlap_ratio": float(overlap / pred_union)
        if pred_union > 0
        else 0.0,
        "out_of_boundary_ratio": float(outside / pred_union)
        if pred_union > 0
        else 0.0,
        "node_f1": float(node_f1),
        "edge_overlap": float(edge_overlap),
        "ged": float(ged),
        "pred_edges": len(pred_edges),
        "target_edges": len(target_edges),
    }


__all__ = ["structural_legality_metrics", "wall_junction_summary"]
