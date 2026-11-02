"""Hardened FEVP metric contracts and reference implementations."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import label as component_label
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon
from shapely.ops import unary_union

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.housemind_structure import (
    RoomTypeGraph,
    node_set_f1,
)
from floorflow.evaluation.structural_legality import wall_junction_summary

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONTRACT = ROOT / "configs/fevp_metric_contract_v2.json"


@dataclass(frozen=True)
class FEVPTolerances:
    """Frozen tolerances used by all hardened evaluators."""

    room_match_min_iou: float = 0.5
    room_polygon_hausdorff_px: float = 0.5
    room_min_area_px: float = 4.0
    room_overlap_area_px2: float = 4.0
    edge_contact_px: float = 2.0
    edge_snap_px: float = 1.5
    wall_junction_cluster_px: float = 2.0
    hole_min_area_px: float = 0.25
    changed_pixel_epsilon: float = 0.0


@dataclass(frozen=True)
class RoomInstance:
    """One connected room instance with an explicit semantic label."""

    index: int
    label: str
    mask: np.ndarray
    polygon: Polygon
    area: float


def load_metric_contract(path: str | Path = DEFAULT_CONTRACT) -> dict:
    """Load the machine-readable metric contract."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def tolerances_from_contract(contract: dict) -> FEVPTolerances:
    """Construct evaluator tolerances from the canonical contract file."""

    values = contract["tolerances"]
    return FEVPTolerances(
        room_match_min_iou=float(values["room_match_min_iou"]),
        room_polygon_hausdorff_px=float(values["room_polygon_hausdorff_px"]),
        room_min_area_px=float(values["room_min_area_px"]),
        room_overlap_area_px2=float(values["room_overlap_area_px2"]),
        edge_contact_px=float(values["edge_contact_px"]),
        edge_snap_px=float(values["edge_snap_px"]),
        wall_junction_cluster_px=float(values["wall_junction_cluster_px"]),
        hole_min_area_px=float(values["hole_min_area_px"]),
        changed_pixel_epsilon=float(values["changed_pixel_epsilon"]),
    )


def _mask_polygon(mask: np.ndarray) -> Polygon:
    contours, _ = cv2.findContours(
        np.asarray(mask, dtype=np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return Polygon()
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 3:
        return Polygon()
    polygon = Polygon(contour.reshape(-1, 2))
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.geom_type == "MultiPolygon":
        polygon = max(polygon.geoms, key=lambda item: item.area)
    if polygon.geom_type != "Polygon":
        return Polygon()
    return polygon


def room_instances(
    labels: np.ndarray,
    taxonomy: FloorplanTaxonomy,
    *,
    min_area: float,
) -> list[RoomInstance]:
    """Resolve connected components for every legal room type."""

    values = np.asarray(labels)
    instances: list[RoomInstance] = []
    for taxonomy_id in taxonomy.room_ids:
        components, count = component_label(values == int(taxonomy_id))
        for component_id in range(1, count + 1):
            mask = components == component_id
            area = float(mask.sum())
            if area < min_area:
                continue
            instances.append(
                RoomInstance(
                    index=len(instances),
                    label=taxonomy.class_names[int(taxonomy_id)],
                    mask=mask,
                    polygon=_mask_polygon(mask),
                    area=area,
                )
            )
    return instances


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    union = np.logical_or(left, right).sum()
    return float(np.logical_and(left, right).sum() / union) if union else 0.0


def match_room_instances(
    before: Sequence[RoomInstance],
    after: Sequence[RoomInstance],
    tolerances: FEVPTolerances,
) -> dict[int, int]:
    """One-to-one semantic room matching with no many-to-one assignment."""

    if not before or not after:
        return {}
    similarity = np.zeros((len(before), len(after)), dtype=np.float64)
    for left_index, left in enumerate(before):
        for right_index, right in enumerate(after):
            if left.label == right.label:
                similarity[left_index, right_index] = _iou(left.mask, right.mask)
    rows, columns = linear_sum_assignment(-similarity)
    return {
        int(row): int(column)
        for row, column in zip(rows, columns, strict=True)
        if similarity[row, column] >= tolerances.room_match_min_iou
    }


def _instance_edges(
    instances: Sequence[RoomInstance],
    tolerance: float,
) -> set[tuple[int, int]]:
    edges = set()
    for left in range(len(instances)):
        for right in range(left + 1, len(instances)):
            if (
                instances[left].polygon.distance(instances[right].polygon)
                <= tolerance
            ):
                edges.add((left, right))
    return edges


def outside_layer_metrics(
    before_labels: np.ndarray,
    after_labels: np.ndarray,
    changed_mask: np.ndarray,
    edit_region: np.ndarray,
    footprint: np.ndarray,
    taxonomy: FloorplanTaxonomy,
    *,
    before_pixels: np.ndarray | None = None,
    after_pixels: np.ndarray | None = None,
    valid_outside: np.ndarray | None = None,
    tolerances: FEVPTolerances,
) -> dict[str, float | int | bool]:
    """Score pixel, room-instance, and typed-edge outside preservation."""

    before = np.asarray(before_labels)
    after = np.asarray(after_labels)
    changed = np.asarray(changed_mask, dtype=bool)
    region = np.asarray(edit_region, dtype=bool)
    support = np.asarray(footprint, dtype=bool)
    if not (
        before.shape == after.shape == changed.shape == region.shape == support.shape
    ):
        raise ValueError("outside metric arrays must share one shape")
    outside = support & ~region if valid_outside is None else np.asarray(valid_outside, bool)
    outside &= support & ~region
    pixel_before = before if before_pixels is None else np.asarray(before_pixels)
    pixel_after = after if after_pixels is None else np.asarray(after_pixels)
    if pixel_before.shape != before.shape or pixel_after.shape != after.shape:
        raise ValueError("pixel arrays must match the canonical label shape")
    outside_pixels = int(outside.sum())
    outside_changes = int(
        np.count_nonzero(pixel_after[outside] != pixel_before[outside])
    )
    before_instances = room_instances(
        before,
        taxonomy,
        min_area=tolerances.room_min_area_px,
    )
    after_instances = room_instances(
        after,
        taxonomy,
        min_area=tolerances.room_min_area_px,
    )
    matches = match_room_instances(before_instances, after_instances, tolerances)
    changed_rooms = {
        instance.index
        for instance in before_instances
        if np.count_nonzero(instance.mask & changed)
        > tolerances.changed_pixel_epsilon
    }
    region_rooms = {
        instance.index
        for instance in before_instances
        if np.count_nonzero(instance.mask & region) > 0
    }
    affected = changed_rooms | region_rooms
    untouched = [
        instance.index
        for instance in before_instances
        if instance.index not in affected
    ]
    matched = [index for index in untouched if index in matches]
    label_exact = len(matched) == len(untouched)
    mask_exact = len(matched) == len(untouched)
    polygon_exact = len(matched) == len(untouched)
    deviations = []
    for index in matched:
        left = before_instances[index]
        right = after_instances[matches[index]]
        mask_exact &= np.array_equal(left.mask, right.mask)
        if np.array_equal(left.mask, right.mask):
            deviation = 0.0
        elif not left.polygon.is_empty and not right.polygon.is_empty:
            deviation = left.polygon.exterior.hausdorff_distance(
                right.polygon.exterior
            )
        else:
            deviation = float("inf")
        deviations.append(float(deviation))
        polygon_exact &= (
            deviation <= tolerances.room_polygon_hausdorff_px + 1.0e-9
        )
    before_edges = {
        edge
        for edge in _instance_edges(
            before_instances,
            tolerances.edge_contact_px,
        )
        if edge[0] in untouched and edge[1] in untouched
    }
    mapped_untouched = {matches[index] for index in matched}
    after_edges = {
        edge
        for edge in _instance_edges(
            after_instances,
            tolerances.edge_contact_px,
        )
        if edge[0] in mapped_untouched and edge[1] in mapped_untouched
    }
    mapped_edges = {
        tuple(sorted((matches[left], matches[right])))
        for left, right in before_edges
        if left in matches and right in matches
    }
    edge_exact = mapped_edges == after_edges
    room_level_exact = bool(
        label_exact
        and mask_exact
        and polygon_exact
        and edge_exact
    )
    return {
        "outside_pixel_exact": (
            float(1.0 - outside_changes / outside_pixels)
            if outside_pixels
            else 1.0
        ),
        "outside_room_level_exact": float(room_level_exact),
        "outside_room_label_exact": float(label_exact),
        "outside_room_mask_exact": float(mask_exact),
        "outside_room_polygon_exact": float(polygon_exact),
        "outside_edge_exact": float(edge_exact),
        "outside_room_polygon_max_deviation_px": (
            max(deviations) if deviations else 0.0
        ),
        "locality_violation_rate": (
            float(outside_changes / outside_pixels) if outside_pixels else 0.0
        ),
        "outside_valid_pixels": outside_pixels,
        "outside_changed_pixels": outside_changes,
        "before_room_instances": len(before_instances),
        "unaffected_room_instances": len(untouched),
        "matched_unaffected_room_instances": len(matched),
        "affected_room_instances": len(affected),
        "changed_room_instances": len(changed_rooms),
        "region_overlap_room_instances": len(region_rooms),
        "unaffected_edge_count": len(mapped_edges),
    }


def validity_checks(
    plan,
    footprint: np.ndarray,
    taxonomy: FloorplanTaxonomy,
    *,
    tolerances: FEVPTolerances,
    room_labels: np.ndarray | None = None,
) -> dict[str, float | bool | list[str]]:
    """Evaluate the five mandatory validity checks independently."""

    errors: list[str] = []
    support = np.asarray(footprint, dtype=np.uint8)
    contours, _ = cv2.findContours(
        support,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    outer_closed = bool(
        contours
        and len(max(contours, key=cv2.contourArea)) >= 4
        and cv2.contourArea(max(contours, key=cv2.contourArea))
        >= tolerances.room_min_area_px
    )
    if not outer_closed:
        errors.append("outer_boundary_open")
    polygons = []
    raw_validity = []
    for room in plan.rooms:
        coordinates = room.get("polygon") or []
        raw_polygon = (
            Polygon(coordinates, room.get("holes", []))
            if len(coordinates) >= 3
            else Polygon()
        )
        raw_validity.append(raw_polygon.is_valid)
        polygons.append(
            raw_polygon.buffer(0)
            if not raw_polygon.is_empty
            else Polygon()
        )
    if room_labels is None:
        labels = [str(room.get("label")) for room in plan.rooms]
    else:
        instances = room_instances(
            room_labels,
            taxonomy,
            min_area=tolerances.room_min_area_px,
        )
        labels = [instance.label for instance in instances]
    room_closed = bool(polygons) and all(
        not polygon.is_empty
        and polygon.is_valid
        and polygon.area >= tolerances.room_min_area_px
        for polygon in polygons
    )
    if not room_closed:
        errors.append("room_polygon_open_or_degenerate")
    no_self_intersection = bool(raw_validity) and all(raw_validity)
    if not no_self_intersection:
        errors.append("room_self_intersection")
    no_overlap = True
    for left in range(len(polygons)):
        for right in range(left + 1, len(polygons)):
            if (
                polygons[left].intersection(polygons[right]).area
                > tolerances.room_overlap_area_px2
            ):
                no_overlap = False
                break
        if not no_overlap:
            break
    if not no_overlap:
        errors.append("room_interior_overlap")
    legal_labels = set(taxonomy.class_names)
    semantic_complete = bool(labels) and all(label in legal_labels for label in labels)
    if not semantic_complete:
        errors.append("illegal_or_missing_room_label")
    valid = bool(
        outer_closed
        and room_closed
        and no_self_intersection
        and no_overlap
        and semantic_complete
    )
    return {
        "validity": float(valid),
        "outer_boundary_closed": float(outer_closed),
        "room_polygon_closed": float(room_closed),
        "no_self_intersection": float(no_self_intersection),
        "no_room_overlap": float(no_overlap),
        "semantic_complete": float(semantic_complete),
        "validity_errors": errors,
    }


def watertight_diagnostics(
    plan,
    *,
    taxonomy: FloorplanTaxonomy | None = None,
    tolerances: FEVPTolerances,
    fpeval_watertight: bool | None = None,
    labels: np.ndarray | None = None,
    footprint: np.ndarray | None = None,
) -> dict[str, float | int | bool | list[str]]:
    """Report watertightness with hole and dangling-endpoint localization."""

    polygons = []
    for room in plan.rooms:
        coordinates = room.get("polygon") or []
        if len(coordinates) < 3:
            continue
        polygon = Polygon(
            coordinates,
            room.get("holes", []),
        ).buffer(0)
        if not polygon.is_empty and polygon.area >= tolerances.room_min_area_px:
            polygons.append(polygon)
    unions = unary_union(polygons) if polygons else Polygon()
    components = list(unions.geoms) if hasattr(unions, "geoms") else [unions]
    geometry_hole_areas = [
        Polygon(interior).area
        for component in components
        if not component.is_empty and component.geom_type == "Polygon"
        for interior in component.interiors
    ]
    raster_hole_areas: list[float] = []
    if labels is not None and taxonomy is not None:
        room_mask = np.isin(
            np.asarray(labels),
            tuple(taxonomy.room_ids),
        )
        if footprint is not None:
            room_mask &= np.asarray(footprint, dtype=bool)
        hole_mask = binary_fill_holes(room_mask) & ~room_mask
        hole_components, count = component_label(hole_mask)
        for component in range(1, count + 1):
            area = float((hole_components == component).sum())
            if area > tolerances.hole_min_area_px:
                raster_hole_areas.append(area)
    hole_areas = (
        raster_hole_areas
        if labels is not None and taxonomy is not None
        else geometry_hole_areas
    )
    holes = [
        area for area in hole_areas if area > tolerances.hole_min_area_px
    ]
    junction = wall_junction_summary(
        plan.walls,
        tolerance=tolerances.wall_junction_cluster_px,
    )
    endpoint_count = 2 * len(plan.walls)
    dangling_rate = (
        float(junction["open_endpoint_count"] / endpoint_count)
        if endpoint_count
        else 0.0
    )
    union_area = float(unions.area) if not unions.is_empty else 0.0
    issue_codes = []
    if not polygons:
        issue_codes.append("no_room_polygons")
    if holes:
        issue_codes.append("interior_hole")
    if len([item for item in components if not item.is_empty]) > 1:
        issue_codes.append("disconnected_components")
    if dangling_rate > 0.0:
        issue_codes.append("dangling_endpoints")
    if fpeval_watertight is False:
        issue_codes.append("fpeval_not_watertight")
    inferred = bool(
        polygons
        and not holes
        and len([item for item in components if not item.is_empty]) == 1
        and (fpeval_watertight is not False)
    )
    return {
        "watertight": inferred,
        "watertight_rate": float(inferred),
        "hole_count": len(holes),
        "hole_area_px": float(sum(holes)),
        "hole_area_ratio": float(sum(holes) / union_area) if union_area else 0.0,
        "disconnected_component_count": max(
            0,
            len([item for item in components if not item.is_empty]) - 1,
        ),
        "dangling_endpoint_count": int(junction["open_endpoint_count"]),
        "dangling_endpoint_rate": dangling_rate,
        "watertight_issue_codes": issue_codes,
    }


def edge_precision_recall_f1(
    predicted: RoomTypeGraph,
    target: RoomTypeGraph,
) -> dict[str, float | int]:
    """Return edge precision, recall, F1, and Jaccard overlap."""

    intersection = len(predicted.edges & target.edges)
    precision = (
        intersection / len(predicted.edges) if predicted.edges else 0.0
    )
    recall = intersection / len(target.edges) if target.edges else 0.0
    denominator = precision + recall
    f1 = (
        2.0 * precision * recall / denominator
        if denominator
        else (1.0 if not predicted.edges and not target.edges else 0.0)
    )
    union = predicted.edges | target.edges
    return {
        "intersection": intersection,
        "predicted_edges": len(predicted.edges),
        "target_edges": len(target.edges),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "overlap": float(intersection / len(union)) if union else 1.0,
    }


def node_precision_recall_f1(
    predicted: RoomTypeGraph,
    target: RoomTypeGraph,
) -> dict[str, float | int]:
    """Expose the reference node matching exactly as precision/recall/F1."""

    return node_set_f1(predicted, target)


__all__ = [
    "DEFAULT_CONTRACT",
    "FEVPTolerances",
    "RoomInstance",
    "edge_precision_recall_f1",
    "load_metric_contract",
    "match_room_instances",
    "node_precision_recall_f1",
    "outside_layer_metrics",
    "room_instances",
    "tolerances_from_contract",
    "validity_checks",
    "watertight_diagnostics",
]
