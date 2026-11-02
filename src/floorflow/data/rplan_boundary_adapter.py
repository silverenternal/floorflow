"""Target-free RPLAN boundary adapter for unified vector floorplan generation."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from shapely.geometry import Polygon
from torch.utils.data import Dataset

from floorflow.data.floorplan_token_targets import token_batch_from_plan
from floorflow.data.rplan_collate import collate_rplan_items
from floorflow.data.rplan_room_limits import limit_canonical_plan_rooms
from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.canonical_floorplan import (
    CanonicalFloorplan,
)
from floorflow.objectives.token_grammar import FloorplanTokenVocabulary
from floorflow.representation.vector_floorplan_assembly import (
    assemble_vector_floorplan,
)

RPLAN_CATEGORY_TO_ROOM = {
    0: "room:LivingRoom",
    1: "room:Bedroom",
    2: "room:Kitchen",
    3: "room:Bath",
    4: "room:Balcony",
    5: "room:Storage",
}


def rplan_taxonomy() -> FloorplanTaxonomy:
    """Return the fixed semantic vocabulary used by the RPLAN adapter."""

    return FloorplanTaxonomy(
        class_names=(
            "background",
            "wall",
            "structure",
            "door:Door",
            "room:LivingRoom",
            "room:Bedroom",
            "room:Kitchen",
            "room:Bath",
            "room:Balcony",
            "room:Storage",
        ),
        room_map={},
        furniture_map={},
        door_map={},
        window_map={},
        source_counts={},
    )


def _finite_ring(
    points: Sequence[Sequence[float]],
    name: str,
) -> tuple[tuple[float, float], ...]:
    ring = []
    for point in points:
        values = tuple(float(value) for value in point[:2])
        if len(values) != 2 or not all(math.isfinite(value) for value in values):
            raise ValueError(f"{name} contains a non-finite point")
        ring.append(values)
    if len(ring) >= 2 and ring[0] == ring[-1]:
        ring.pop()
    if len(ring) < 3:
        raise ValueError(f"{name} needs at least three vertices")
    return tuple(ring)


def _room_ring(room: dict) -> tuple[tuple[float, float], ...]:
    polygon = room.get("r_boundary") or []
    if len(polygon) < 3:
        box = room.get("box") or []
        if len(box) != 4:
            raise ValueError("room has neither boundary nor box geometry")
        x0, y0, x1, y1 = (float(value) for value in box)
        polygon = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
    ring = _finite_ring(polygon, f"room {room.get('id')}")
    shape = Polygon(ring)
    if not shape.is_valid or shape.area <= 0.0:
        raise ValueError(f"room {room.get('id')} has invalid polygon geometry")
    return ring


def _source_hash(row: dict) -> str:
    payload = json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def rplan_row_to_canonical(
    row: dict,
    taxonomy: FloorplanTaxonomy,
    *,
    image_size: int = 256,
) -> CanonicalFloorplan:
    """Convert one DiffPlanner/CE2EPlan RPLAN row without reading target pixels."""

    boundary = _finite_ring(
        [(point[0], point[1]) for point in row["boundary"]],
        "boundary",
    )
    footprint_shape = Polygon(boundary)
    if not footprint_shape.is_valid or footprint_shape.area <= 0.0:
        raise ValueError("RPLAN boundary is invalid")
    source_rooms = sorted(
        row.get("rooms", []),
        key=lambda room: (
            int(room.get("category", -1)),
            float((room.get("location") or (0.0, 0.0))[1]),
            float((room.get("location") or (0.0, 0.0))[0]),
            int(room.get("id", -1)),
        ),
    )
    if not source_rooms:
        raise ValueError("RPLAN case has no rooms")
    room_ids = {}
    room_rows = []
    for index, source in enumerate(source_rooms, start=1):
        category = int(source.get("category", -1))
        label = RPLAN_CATEGORY_TO_ROOM.get(category)
        if label is None or not taxonomy.is_room(label):
            raise ValueError(f"unknown RPLAN room category: {category}")
        ring = _room_ring(source)
        room_id = f"room_{index:02d}"
        source_id = int(source.get("id", index - 1))
        if source_id in room_ids:
            raise ValueError(f"duplicate RPLAN room id: {source_id}")
        room_ids[source_id] = room_id
        room_rows.append(
            {"id": room_id, "label": label, "polygon": ring, "holes": []}
        )
    adjacency = []
    for edge in row.get("adjacencies", []):
        if len(edge) != 2:
            continue
        left = room_ids.get(int(edge[0]))
        right = room_ids.get(int(edge[1]))
        if left is not None and right is not None and left != right:
            adjacency.append(tuple(sorted((left, right))))
    return assemble_vector_floorplan(
        case_id=str(row["name"]),
        track="rplan_generation",
        width=int(image_size),
        height=int(image_size),
        footprint=boundary,
        rooms=room_rows,
        adjacency=tuple(sorted(set(adjacency))),
        source_hash=_source_hash(row),
        taxonomy=taxonomy,
    )


def resample_ring(
    ring: Sequence[Sequence[float]],
    point_count: int,
) -> np.ndarray:
    """Sample a canonical polygon perimeter into a fixed ordered loop."""

    if point_count < 3:
        raise ValueError("point_count must be at least three")
    points = np.asarray(ring, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        raise ValueError("ring must be [P,2] with at least three points")
    if np.allclose(points[0], points[-1]):
        points = points[:-1]
    area = np.sum(
        points[:, 0] * np.roll(points[:, 1], -1)
        - np.roll(points[:, 0], -1) * points[:, 1]
    )
    if abs(float(area)) <= 1.0e-9:
        raise ValueError("ring has zero area")
    if area < 0:
        points = points[::-1]
    start = int(np.lexsort((points[:, 0], points[:, 1]))[0])
    points = np.roll(points, -start, axis=0)
    closed = np.concatenate([points, points[:1]], axis=0)
    lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    samples = np.linspace(
        0.0,
        cumulative[-1],
        int(point_count),
        endpoint=False,
    )
    indices = np.searchsorted(cumulative, samples, side="right") - 1
    indices = np.clip(indices, 0, len(lengths) - 1)
    fraction = (
        (samples - cumulative[indices])
        / np.maximum(lengths[indices], 1.0e-8)
    )
    return closed[indices] + fraction[:, None] * (
        closed[indices + 1] - closed[indices]
    )


def load_rplan_rows(
    path: str | Path,
    *,
    offset: int = 0,
    limit: int | None = None,
) -> list[dict]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    end = None if limit is None else int(offset) + int(limit)
    return rows[int(offset) : end]


def entrance_segment_from_row(
    row: dict,
    *,
    canvas_size: int = 256,
) -> tuple[np.ndarray, bool]:
    """Recover the conditioned opening interval from an entrance rectangle."""

    raw = row.get("entrance_expand")
    if not isinstance(raw, list) or len(raw) < 2:
        return np.zeros((2, 2), dtype=np.float32), False
    points = np.asarray(raw, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        return np.zeros((2, 2), dtype=np.float32), False
    center = points.mean(axis=0)
    centered = points - center
    covariance = centered.T @ centered
    _values, vectors = np.linalg.eigh(covariance)
    rectangle_tangent = vectors[:, -1]
    boundary_raw = row.get("boundary")
    if not isinstance(boundary_raw, list) or len(boundary_raw) < 3:
        return np.zeros((2, 2), dtype=np.float32), False
    boundary = np.asarray([point[:2] for point in boundary_raw], dtype=np.float64)
    if boundary.ndim != 2 or boundary.shape[1] != 2:
        return np.zeros((2, 2), dtype=np.float32), False
    edges = np.roll(boundary, -1, axis=0) - boundary
    edge_lengths = np.linalg.norm(edges, axis=1)
    candidates = []
    for index, (edge, edge_length) in enumerate(zip(edges, edge_lengths, strict=True)):
        if edge_length <= 1.0e-6:
            continue
        direction = edge / edge_length
        projection = np.clip((center - boundary[index]) @ direction, 0.0, edge_length)
        closest = boundary[index] + direction * projection
        candidates.append(
            (
                abs(float(direction @ rectangle_tangent)),
                float(np.linalg.norm(closest - center)),
                closest,
                direction,
                edge_length,
                projection,
            )
        )
    if not candidates:
        return np.zeros((2, 2), dtype=np.float32), False
    _compatibility, _distance, closest, direction, edge_length, _projection = min(
        candidates,
        key=lambda item: (-item[0], item[1]),
    )
    span = float(np.ptp(centered @ direction))
    half_span = 0.5 * min(span, edge_length)
    start = closest - direction * half_span
    end = closest + direction * half_span
    if np.linalg.norm(end - start) <= 1.0e-6:
        return np.zeros((2, 2), dtype=np.float32), False
    normalized = np.stack([start, end]) / max(1, int(canvas_size) - 1)
    return normalized.astype(np.float32), True


@dataclass(frozen=True)
class RPlanTensorLimits:
    image_size: int = 128
    canvas_size: int = 256
    max_boundary_points: int = 64
    max_rooms: int = 12
    room_points: int = 24
    max_walls: int = 128
    max_openings: int = 8
    max_junctions: int = 32


class RPlanBoundaryDataset(Dataset):
    """RPLAN rows rendered into target-free boundary conditioning tensors."""

    def __init__(
        self,
        path: str | Path,
        taxonomy: FloorplanTaxonomy,
        *,
        offset: int = 0,
        limit: int | None = None,
        limits: RPlanTensorLimits | None = None,
        rows: list[dict] | None = None,
        token_vocabulary: FloorplanTokenVocabulary | None = None,
        token_max_length: int = 512,
        token_numeric_bin_size: int = 8,
        token_geometry_mode: str = "reference",
    ) -> None:
        self.rows = (
            rows
            if rows is not None
            else load_rplan_rows(path, offset=offset, limit=limit)
        )
        self.taxonomy = taxonomy
        self.limits = limits or RPlanTensorLimits()
        self.token_vocabulary = token_vocabulary
        self.token_max_length = int(token_max_length)
        self.token_numeric_bin_size = int(token_numeric_bin_size)
        self.token_geometry_mode = str(token_geometry_mode)
        self._plan_cache: dict[int, CanonicalFloorplan] = {}
        self._token_target_cache: dict[int, dict[str, Any]] = {}
        self._cache_limit = 512

    def _plan(self, index: int) -> CanonicalFloorplan:
        cached = self._plan_cache.get(index)
        if cached is not None:
            return cached
        plan = rplan_row_to_canonical(
            self.rows[index],
            self.taxonomy,
            image_size=self.limits.canvas_size,
        )
        if len(self.rows) <= self._cache_limit:
            self._plan_cache[index] = plan
        return plan

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        plan = self._plan(index)
        item = canonical_plan_to_tensor_item(
            plan,
            self.taxonomy,
            self.limits,
            case_name=plan.case_id,
        )
        entrance, entrance_valid = entrance_segment_from_row(
            self.rows[index],
            canvas_size=self.limits.canvas_size,
        )
        item["entrance_points"] = torch.from_numpy(entrance)
        item["entrance_valid"] = torch.tensor(entrance_valid, dtype=torch.bool)
        if self.token_vocabulary is not None:
            token_item = self._token_target_cache.pop(index, None)
            if token_item is None:
                token_item = token_batch_from_plan(
                    limit_canonical_plan_rooms(plan, self.limits.max_rooms),
                    self.taxonomy,
                    self.token_vocabulary,
                    max_length=self.token_max_length,
                    numeric_bin_size=self.token_numeric_bin_size,
                    geometry_token_mode=self.token_geometry_mode,
                ).to_item()
            item.update(token_item)
        return item

    def token_length(self, index: int) -> int:
        if self.token_vocabulary is None:
            raise ValueError("token length requires an enabled token vocabulary")
        plan = self._plan(index)
        target = token_batch_from_plan(
                limit_canonical_plan_rooms(plan, self.limits.max_rooms),
                self.taxonomy,
                self.token_vocabulary,
                max_length=self.token_max_length,
                numeric_bin_size=self.token_numeric_bin_size,
                geometry_token_mode=self.token_geometry_mode,
            )
        self._token_target_cache[index] = target.to_item()
        return int(target.length)


def _fixed_points(
    points: Sequence[Sequence[float]],
    maximum: int,
    canvas: int,
) -> tuple[np.ndarray, np.ndarray]:
    output = np.zeros((maximum, 2), dtype=np.float32)
    valid = np.zeros(maximum, dtype=bool)
    values = np.asarray(points, dtype=np.float64)
    count = min(len(values), maximum)
    if len(values) > maximum:
        values = resample_ring(values, maximum)
        count = maximum
    output[:count] = values[:count].astype(np.float32) / max(
        1,
        canvas - 1,
    )
    valid[:count] = True
    return output, valid


def canonical_plan_to_tensor_item(
    plan: CanonicalFloorplan,
    taxonomy: FloorplanTaxonomy,
    limits: RPlanTensorLimits,
    *,
    case_name: str,
) -> dict[str, Any]:
    """Convert a decoded canonical plan into supervised model tensors."""

    boundary, boundary_valid = _fixed_points(
        plan.footprint,
        limits.max_boundary_points,
        limits.canvas_size,
    )
    boundary_mask = np.zeros(
        (1, limits.image_size, limits.image_size),
        dtype=np.float32,
    )
    boundary_pixels = np.rint(
        np.asarray(plan.footprint)
        * (limits.image_size - 1)
        / max(1, limits.canvas_size - 1)
    ).astype(np.int32)
    cv2.fillPoly(boundary_mask[0], [boundary_pixels], 1.0)
    room_classes = np.zeros(limits.max_rooms, dtype=np.int64)
    room_exists = np.zeros(limits.max_rooms, dtype=bool)
    room_polygons = np.zeros(
        (limits.max_rooms, limits.room_points, 2),
        dtype=np.float32,
    )
    room_point_valid = np.zeros(
        (limits.max_rooms, limits.room_points),
        dtype=bool,
    )
    room_masks = np.zeros(
        (limits.max_rooms, limits.image_size, limits.image_size),
        dtype=np.float32,
    )
    target_labels = np.zeros(
        (limits.image_size, limits.image_size),
        dtype=np.int32,
    )
    room_target_labels = np.zeros_like(target_labels)
    room_target_mask = np.zeros_like(target_labels, dtype=np.float32)
    room_lookup = {
        taxonomy.class_names[class_id]: index
        for index, class_id in enumerate(taxonomy.room_ids)
    }
    target_rooms = limit_canonical_plan_rooms(
        plan,
        limits.max_rooms,
    ).rooms
    for room_index, room in enumerate(target_rooms):
        polygon = np.asarray(room.polygon, dtype=np.float64)
        if len(polygon) > limits.room_points:
            polygon = resample_ring(polygon, limits.room_points)
        point_count = min(len(polygon), limits.room_points)
        normalized = (
            polygon / max(1, limits.canvas_size - 1)
        ).astype(np.float32)
        room_polygons[room_index, :point_count] = normalized[:point_count]
        room_polygons[room_index, point_count:] = normalized[point_count - 1]
        room_point_valid[room_index, :point_count] = True
        room_classes[room_index] = room_lookup[room.label]
        room_exists[room_index] = True
        pixels = np.rint(
            np.asarray(room.polygon)
            * (limits.image_size - 1)
            / max(1, limits.canvas_size - 1)
        ).astype(np.int32)
        cv2.fillPoly(room_masks[room_index], [pixels], 1.0)
        cv2.fillPoly(
            target_labels,
            [pixels],
            taxonomy.class_id(room.label),
        )
        cv2.fillPoly(
            room_target_labels,
            [pixels],
            room_lookup[room.label],
        )
        cv2.fillPoly(room_target_mask, [pixels], 1.0)
    room_id_index = {
        room.id: index for index, room in enumerate(target_rooms)
    }
    adjacency = np.zeros(
        (limits.max_rooms, limits.max_rooms),
        dtype=np.float32,
    )
    for left, right in plan.adjacency:
        if left in room_id_index and right in room_id_index:
            left_index = room_id_index[left]
            right_index = room_id_index[right]
            adjacency[left_index, right_index] = 1.0
            adjacency[right_index, left_index] = 1.0
    wall_endpoints = np.zeros(
        (limits.max_walls, 2, 2),
        dtype=np.float32,
    )
    wall_valid = np.zeros(limits.max_walls, dtype=bool)
    wall_exterior = np.zeros(limits.max_walls, dtype=np.float32)
    wall_ids = {}
    for wall_index, wall in enumerate(plan.walls[: limits.max_walls]):
        wall_endpoints[wall_index] = np.asarray(
            [wall.start, wall.end],
            dtype=np.float32,
        ) / max(1, limits.canvas_size - 1)
        wall_valid[wall_index] = True
        wall_exterior[wall_index] = float(wall.wall_type == "exterior")
        wall_ids[wall.id] = wall_index
    opening_center = np.zeros((limits.max_openings, 2), dtype=np.float32)
    opening_width = np.zeros(limits.max_openings, dtype=np.float32)
    opening_host = np.full(limits.max_openings, -100, dtype=np.int64)
    opening_valid = np.zeros(limits.max_openings, dtype=bool)
    for opening_index, opening in enumerate(
        plan.openings[: limits.max_openings]
    ):
        opening_center[opening_index] = np.asarray(
            opening.center,
            dtype=np.float32,
        ) / max(1, limits.canvas_size - 1)
        opening_width[opening_index] = opening.width / max(
            1,
            limits.canvas_size - 1,
        )
        host_index = wall_ids.get(opening.wall_id, -100)
        opening_host[opening_index] = host_index
        opening_valid[opening_index] = int(host_index) >= 0
    junction_points = np.zeros(
        (limits.max_junctions, 2),
        dtype=np.float32,
    )
    junction_degree = np.zeros(limits.max_junctions, dtype=np.int64)
    junction_valid = np.zeros(limits.max_junctions, dtype=bool)
    for junction_index, junction in enumerate(plan.junctions[: limits.max_junctions]):
        junction_points[junction_index] = np.asarray(
            junction.point,
            dtype=np.float32,
        ) / max(1, limits.canvas_size - 1)
        junction_degree[junction_index] = int(junction.degree)
        junction_valid[junction_index] = True
    return {
        "case_name": case_name,
        "boundary_points": torch.from_numpy(boundary),
        "boundary_valid": torch.from_numpy(boundary_valid),
        "boundary_mask": torch.from_numpy(boundary_mask),
        "room_classes": torch.from_numpy(room_classes),
        "room_exists": torch.from_numpy(room_exists),
        "room_polygons": torch.from_numpy(room_polygons),
        "room_point_valid": torch.from_numpy(room_point_valid),
        "room_masks": torch.from_numpy(room_masks),
        "target_labels": torch.from_numpy(target_labels.astype(np.int64)),
        "room_target_labels": torch.from_numpy(
            room_target_labels.astype(np.int64)
        ),
        "room_target_mask": torch.from_numpy(room_target_mask),
        "room_count": torch.tensor(len(target_rooms), dtype=torch.long),
        "source_room_count": torch.tensor(
            len(plan.rooms),
            dtype=torch.long,
        ),
        "adjacency": torch.from_numpy(adjacency),
        "wall_endpoints": torch.from_numpy(wall_endpoints),
        "wall_valid": torch.from_numpy(wall_valid),
        "wall_exterior": torch.from_numpy(wall_exterior),
        "opening_center": torch.from_numpy(opening_center),
        "opening_width": torch.from_numpy(opening_width),
        "opening_host": torch.from_numpy(opening_host),
        "opening_valid": torch.from_numpy(opening_valid),
        "junction_points": torch.from_numpy(junction_points),
        "junction_degree": torch.from_numpy(junction_degree),
        "junction_valid": torch.from_numpy(junction_valid),
    }


__all__ = [
    "RPLAN_CATEGORY_TO_ROOM",
    "RPlanBoundaryDataset",
    "RPlanTensorLimits",
    "canonical_plan_to_tensor_item",
    "collate_rplan_items",
    "entrance_segment_from_row",
    "limit_canonical_plan_rooms",
    "load_rplan_rows",
    "resample_ring",
    "rplan_row_to_canonical",
    "rplan_taxonomy",
]
