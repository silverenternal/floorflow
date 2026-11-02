"""Conservative geometry refinement for Raster2Seq polygon predictions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapely.geometry import Polygon

OPENING_TYPES = frozenset({9, 10})


@dataclass(frozen=True)
class Raster2SeqRefineConfig:
    """A deterministic postprocessor selected without test-set feedback."""

    simplify_tolerance: float = 0.0
    snap_tolerance: float = 0.0
    duplicate_iou: float = 1.01
    minimum_room_area: float = 0.0


def _room_points(item: dict) -> np.ndarray:
    points = np.asarray(item["segmentation"], dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("room segmentation must be an Nx2 point sequence")
    if len(points) > 1 and np.allclose(points[0], points[-1]):
        points = points[:-1]
    return points


def _simplify_points(points: np.ndarray, tolerance: float) -> np.ndarray:
    if len(points) < 3:
        return points
    output = []
    for point in points:
        if not output or np.linalg.norm(point - output[-1]) > tolerance:
            output.append(point)
    if len(output) > 1 and np.linalg.norm(output[0] - output[-1]) <= tolerance:
        output.pop()
    if len(output) < 3:
        return np.asarray(output)
    simplified = []
    count = len(output)
    for index, point in enumerate(output):
        left = np.asarray(output[(index - 1) % count])
        right = np.asarray(output[(index + 1) % count])
        first = np.asarray(point) - left
        second = right - np.asarray(point)
        cross = abs(float(first[0] * second[1] - first[1] * second[0]))
        if cross > tolerance * max(1.0, float(np.linalg.norm(right - left))):
            simplified.append(point)
    return np.asarray(simplified) if len(simplified) >= 3 else np.asarray(output)


def _cluster_axis(values: np.ndarray, tolerance: float) -> dict[float, float]:
    if tolerance <= 0.0 or not len(values):
        return {}
    ordered = np.sort(np.unique(values))
    groups = [[float(ordered[0])]]
    for value in ordered[1:]:
        if float(value) - groups[-1][-1] <= tolerance:
            groups[-1].append(float(value))
        else:
            groups.append([float(value)])
    mapping = {}
    for group in groups:
        center = float(np.mean(group))
        mapping.update({value: center for value in group})
    return mapping


def _snap_rooms(rooms: list[dict], tolerance: float) -> list[dict]:
    if tolerance <= 0.0 or not rooms:
        return rooms
    points = [_room_points(room) for room in rooms]
    x_map = _cluster_axis(np.concatenate([item[:, 0] for item in points]), tolerance)
    y_map = _cluster_axis(np.concatenate([item[:, 1] for item in points]), tolerance)
    output = []
    for room, values in zip(rooms, points, strict=True):
        snapped = values.copy()
        snapped[:, 0] = [x_map[float(value)] for value in values[:, 0]]
        snapped[:, 1] = [y_map[float(value)] for value in values[:, 1]]
        output.append({**room, "segmentation": snapped.tolist()})
    return output


def _polygon(item: dict) -> Polygon | None:
    try:
        value = Polygon(_room_points(item))
    except (ValueError, TypeError):
        return None
    if not value.is_valid or value.area <= 0.0:
        value = value.buffer(0)
    return value if isinstance(value, Polygon) and value.area > 0.0 else None


def _deduplicate_rooms(rooms: list[dict], threshold: float) -> list[dict]:
    if threshold > 1.0 or len(rooms) < 2:
        return rooms
    def area(item: dict) -> float:
        geometry = _polygon(item)
        return float(geometry.area) if geometry is not None else 0.0

    ranked = sorted(rooms, key=area, reverse=True)
    kept = []
    for room in ranked:
        geometry = _polygon(room)
        if geometry is None:
            continue
        duplicate = False
        for other in kept:
            if int(room["category_id"]) != int(other["category_id"]):
                continue
            other_geometry = _polygon(other)
            if other_geometry is None:
                continue
            union = geometry.union(other_geometry).area
            if union > 0.0 and geometry.intersection(other_geometry).area / union > threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(room)
    return sorted(kept, key=lambda item: int(item["id"]))


def refine_raster2seq_items(
    items: list[dict],
    config: Raster2SeqRefineConfig,
) -> tuple[list[dict], dict[str, int]]:
    """Refine room polygons while preserving opening lines and class ids."""

    if (
        config.simplify_tolerance <= 0.0
        and config.snap_tolerance <= 0.0
        and config.duplicate_iou > 1.0
        and config.minimum_room_area <= 0.0
    ):
        rooms = sum(int(item["category_id"]) not in OPENING_TYPES for item in items)
        return list(items), {
            "input_rooms": rooms,
            "output_rooms": rooms,
            "removed_rooms": 0,
            "input_openings": len(items) - rooms,
        }
    rooms = []
    openings = []
    input_rooms = 0
    for item in items:
        if int(item["category_id"]) in OPENING_TYPES:
            openings.append(item)
            continue
        input_rooms += 1
        points = _simplify_points(_room_points(item), config.simplify_tolerance)
        geometry = Polygon(points).buffer(0) if len(points) >= 3 else None
        if geometry is None or not isinstance(geometry, Polygon):
            continue
        if geometry.area < config.minimum_room_area:
            continue
        rooms.append({**item, "segmentation": points.tolist()})
    rooms = _snap_rooms(rooms, config.snap_tolerance)
    rooms = _deduplicate_rooms(rooms, config.duplicate_iou)
    output = sorted(rooms + openings, key=lambda item: int(item["id"]))
    return output, {
        "input_rooms": input_rooms,
        "output_rooms": len(rooms),
        "removed_rooms": input_rooms - len(rooms),
        "input_openings": len(openings),
    }


def _item_iou(left: dict, right: dict) -> float:
    if int(left["category_id"]) != int(right["category_id"]):
        return 0.0
    left_polygon = _polygon(left)
    right_polygon = _polygon(right)
    if left_polygon is None or right_polygon is None:
        return 0.0
    union = left_polygon.union(right_polygon).area
    return (
        left_polygon.intersection(right_polygon).area / union
        if union > 0.0
        else 0.0
    )


def fuse_raster2seq_items(
    primary: list[dict],
    secondary: list[dict],
    *,
    mode: str,
    match_iou: float = 0.5,
) -> tuple[list[dict], dict[str, int]]:
    """Fuse two deterministic prediction sets without case-level selection."""

    if mode == "union":
        ranked = sorted(
            [*primary, *secondary],
            key=lambda item: (
                _polygon(item).area if _polygon(item) is not None else 0.0
            ),
            reverse=True,
        )
        kept = []
        for item in ranked:
            if not any(
                _item_iou(item, other) >= match_iou
                for other in kept
                if int(item["category_id"]) == int(other["category_id"])
            ):
                kept.append(item)
        output = sorted(kept, key=lambda item: int(item["id"]))
    elif mode == "intersection":
        output = []
        for item in primary:
            matches = [
                _item_iou(item, other)
                for other in secondary
                if int(item["category_id"]) == int(other["category_id"])
            ]
            if matches and max(matches) >= match_iou:
                output.append(item)
    else:
        raise ValueError(f"unknown fusion mode: {mode}")
    return output, {
        "primary_items": len(primary),
        "secondary_items": len(secondary),
        "output_items": len(output),
    }


__all__ = [
    "Raster2SeqRefineConfig",
    "fuse_raster2seq_items",
    "refine_raster2seq_items",
]
