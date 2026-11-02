"""Detection-style readout from raster predictions to wall-first plans."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import pairwise

import cv2
import numpy as np
from scipy.ndimage import label as component_label
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union
from skan import Skeleton
from skimage.morphology import skeletonize

from floorflow.data.taxonomy import FloorplanTaxonomy


@dataclass(frozen=True)
class ReadoutWall:
    start: tuple[float, float]
    end: tuple[float, float]
    thickness: float
    wall_type: str = "interior"
    continuity: float = 1.0


@dataclass(frozen=True)
class ReadoutOpening:
    center: tuple[float, float]
    width: float
    kind: str
    wall_index: int


@dataclass(frozen=True)
class ReadoutFurniture:
    """A vector furniture instance attached to one room polygon."""

    label: str
    polygon: tuple[tuple[float, float], ...]
    room_index: int
    confidence: float = 1.0


@dataclass(frozen=True)
class WallSkeletonPolicy:
    """Raster skeleton parameters shared by wall and opening-host readout."""

    close_kernel: int = 0
    simplification_fraction: float = 0.006
    minimum_segment_length: float = 2.5

    def validate(self) -> None:
        if self.close_kernel < 0 or (
            self.close_kernel > 0 and self.close_kernel % 2 == 0
        ):
            raise ValueError(
                "close_kernel must be zero (auto) or a positive odd integer"
            )
        if self.simplification_fraction < 0.0:
            raise ValueError("simplification_fraction must be non-negative")
        if self.minimum_segment_length < 0.0:
            raise ValueError("minimum_segment_length must be non-negative")


@dataclass
class RasterWallPlan:
    frame: float
    walls: list[ReadoutWall]
    rooms: list[dict]
    openings: list[ReadoutOpening]
    room_source: str = "graph_faces"
    furniture: list[ReadoutFurniture] = field(default_factory=list)
    junctions: list[dict] = field(default_factory=list)

    def to_json(self) -> dict:
        wall_rows = []
        for index, wall in enumerate(self.walls):
            wall_rows.append(
                {
                    "id": f"wall_{index + 1}",
                    "start": list(wall.start),
                    "end": list(wall.end),
                    "thickness": wall.thickness,
                    "type": wall.wall_type,
                    "openings": [],
                }
            )
        for opening in self.openings:
            if not 0 <= opening.wall_index < len(wall_rows):
                continue
            wall_rows[opening.wall_index]["openings"].append(
                {
                    "type": opening.kind,
                    "center": list(opening.center),
                    "width": opening.width,
                }
            )
        return {
            "walls": wall_rows,
            "rooms": self.rooms,
            "furniture": [
                {
                    "label": item.label,
                    "polygon": [
                        list(point)
                        for point in item.polygon
                    ],
                    "room_id": f"room_{item.room_index + 1}",
                    "confidence": item.confidence,
                }
                for item in self.furniture
            ],
            "frame": self.frame,
            "room_source": self.room_source,
            "junctions": self.junctions,
        }


def _skeleton_segments(
    mask: np.ndarray,
    *,
    policy: WallSkeletonPolicy | None = None,
) -> tuple[list[ReadoutWall], np.ndarray]:
    policy = policy or WallSkeletonPolicy()
    policy.validate()
    binary = (mask > 0).astype(np.uint8)
    if not binary.any():
        return [], np.zeros((1, 1), dtype=np.uint8)
    kernel_size = (
        policy.close_kernel
        if policy.close_kernel > 0
        else max(3, round(math.sqrt(binary.sum()) / 160.0) | 1)
    )
    closed = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        np.ones((kernel_size, kernel_size), dtype=np.uint8),
    )
    skeleton = skeletonize(closed > 0).astype(np.uint8)
    # skan preserves parallel paths between the same junction pair.  sknw's
    # graph builder can fold a closed room into a self-loop under current
    # scikit-image releases, which prevents polygonization.
    try:
        graph = Skeleton(skeleton)
    except (RuntimeError, ValueError):
        return [], skeleton
    walls: list[ReadoutWall] = []
    epsilon = max(
        1.5,
        max(binary.shape) * policy.simplification_fraction,
    )
    for path in graph.paths_list():
        if len(path) < 2:
            continue
        xy = graph.coordinates[np.asarray(path, dtype=np.int64), ::-1].astype(
            np.float32,
            copy=False,
        )
        closed_path = len(xy) > 2 and np.allclose(
            xy[0],
            xy[-1],
            rtol=0.0,
            atol=1.0e-6,
        )
        simplified = cv2.approxPolyDP(
            xy.reshape(-1, 1, 2),
            epsilon,
            closed_path,
        ).reshape(-1, 2)
        if closed_path and len(simplified) >= 3:
            simplified = np.concatenate([simplified, simplified[:1]], axis=0)
        for start, end in pairwise(simplified):
            length = float(np.linalg.norm(end - start))
            if length < policy.minimum_segment_length:
                continue
            walls.append(
                ReadoutWall(
                    start=(float(start[0]), float(start[1])),
                    end=(float(end[0]), float(end[1])),
                    thickness=3.0,
                    continuity=_segment_continuity(
                        binary,
                        start,
                        end,
                    ),
                )
            )
    return walls, skeleton


def _segment_continuity(
    mask: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    *,
    samples: int = 9,
    radius: int = 1,
) -> float:
    """Fraction of sampled wall points that touch the binary wall mask."""

    height, width = mask.shape
    hits = 0
    for step in range(samples):
        fraction = step / max(1, samples - 1)
        x = start[0] + fraction * (end[0] - start[0])
        y = start[1] + fraction * (end[1] - start[1])
        left = max(0, round(x) - radius)
        right = min(width, round(x) + radius + 1)
        top = max(0, round(y) - radius)
        bottom = min(height, round(y) + radius + 1)
        hits += int(bool(mask[top:bottom, left:right].any()))
    return hits / samples


def _face_label(
    polygon: Polygon,
    room_labels: np.ndarray,
    taxonomy: FloorplanTaxonomy,
) -> tuple[str, float]:
    height, width = room_labels.shape
    x0, y0, x1, y1 = polygon.bounds
    left = max(0, math.floor(x0) - 1)
    top = max(0, math.floor(y0) - 1)
    right = min(width, math.ceil(x1) + 2)
    bottom = min(height, math.ceil(y1) + 2)
    local = np.zeros((max(1, bottom - top), max(1, right - left)), dtype=np.uint8)
    points = np.rint(
        np.asarray(polygon.exterior.coords[:-1], dtype=np.float64)
        - np.asarray([left, top])
    ).astype(np.int32)
    cv2.fillPoly(local, [points], 1)
    values = room_labels[top:bottom, left:right][local.astype(bool)]
    if values.size == 0:
        return "undefined", 0.0
    room_ids = np.asarray(taxonomy.room_ids, dtype=np.int64)
    counts = np.bincount(values.reshape(-1), minlength=taxonomy.num_classes)
    room_counts = counts[room_ids]
    index = int(room_counts.argmax())
    total = int(room_counts.sum())
    if total == 0:
        return "undefined", 0.0
    return taxonomy.class_names[int(room_ids[index])], float(room_counts[index] / total)


def _wall_faces(
    walls: list[ReadoutWall],
    room_labels: np.ndarray,
    taxonomy: FloorplanTaxonomy,
) -> list[dict]:
    if len(walls) < 3:
        return []
    tolerance = max(2.0, max(room_labels.shape) * 0.012)
    endpoints: list[tuple[float, float]] = []
    for wall in walls:
        endpoints.extend([wall.start, wall.end])
    clusters: list[list[tuple[float, float]]] = []
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
    snapped = {}
    for cluster in clusters:
        center = (
            sum(item[0] for item in cluster) / len(cluster),
            sum(item[1] for item in cluster) / len(cluster),
        )
        snapped.update({point: center for point in cluster})
    lines = [
        LineString(
            [
                snapped[wall.start],
                snapped[wall.end],
            ]
        )
        for wall in walls
        if math.dist(wall.start, wall.end) > 1.0e-6
    ]
    if not lines:
        return []
    polygons = list(_polygonize(unary_union(lines)))
    min_area = room_labels.size * 0.0004
    rooms = []
    for polygon in polygons:
        if polygon.is_empty or polygon.area < min_area:
            continue
        label, purity = _face_label(polygon, room_labels, taxonomy)
        points = np.asarray(polygon.exterior.coords[:-1], dtype=np.float64)
        wall_ids = [
            f"wall_{index + 1}"
            for index, wall in enumerate(walls)
            if LineString([wall.start, wall.end]).buffer(2.0).intersects(polygon.boundary)
        ]
        rooms.append(
            {
                "label": label if purity >= 0.3 else "undefined",
                "walls": wall_ids,
                "polygon": points.tolist(),
            }
        )
    return rooms


def _polygonize(geometry):
    from shapely.ops import polygonize

    return polygonize(geometry)


def _opening_components(
    labels: np.ndarray,
    taxonomy: FloorplanTaxonomy,
    walls: list[ReadoutWall],
) -> list[ReadoutOpening]:
    opening_ids = np.asarray(taxonomy.opening_ids, dtype=np.int64)
    opening_mask = np.isin(labels, opening_ids)
    components, count = component_label(opening_mask)
    openings = []
    for component_id in range(1, int(count) + 1):
        ys, xs = np.nonzero(components == component_id)
        if len(xs) < 2:
            continue
        center = (float(xs.mean()), float(ys.mean()))
        values = labels[ys, xs]
        class_id = int(np.bincount(values, minlength=taxonomy.num_classes).argmax())
        width = float(max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1))
        candidates = []
        for index, wall in enumerate(walls):
            line = LineString([wall.start, wall.end])
            distance = line.distance(Point(center))
            candidates.append((distance, index, line))
        if not candidates:
            continue
        feasible = [
            candidate
            for candidate in candidates
            if candidate[2].length >= 0.8 * width
            and candidate[0] <= max(3.0, 0.02 * max(labels.shape))
        ]
        _best_distance, best_index, line = min(
            feasible or candidates,
            key=lambda candidate: candidate[0],
        )
        offset = line.project(Point(center))
        point = line.interpolate(offset)
        projected_center = (float(point.x), float(point.y))
        width = min(width, float(line.length))
        openings.append(
            ReadoutOpening(
                center=projected_center,
                width=width,
                kind=(
                    "window"
                    if taxonomy.class_names[class_id].startswith("window:")
                    else "door"
                ),
                wall_index=best_index,
            )
        )
    return openings


def raster_labels_to_wall_plan(
    labels: np.ndarray,
    taxonomy: FloorplanTaxonomy,
    *,
    room_labels: np.ndarray | None = None,
    wall_policy: WallSkeletonPolicy | None = None,
) -> RasterWallPlan:
    labels = np.asarray(labels, dtype=np.int64)
    semantic_labels = (
        labels if room_labels is None else np.asarray(room_labels, dtype=np.int64)
    )
    if semantic_labels.shape != labels.shape:
        raise ValueError("room_labels and labels must share one shape")
    wall_mask = labels == taxonomy.class_id("wall")
    opening_mask = np.isin(labels, np.asarray(taxonomy.opening_ids))
    walls, _skeleton = _skeleton_segments(
        wall_mask | opening_mask,
        policy=wall_policy,
    )
    rooms = _wall_faces(walls, semantic_labels, taxonomy)
    openings = _opening_components(labels, taxonomy, walls)
    return RasterWallPlan(
        frame=float(max(labels.shape)),
        walls=walls,
        rooms=rooms,
        openings=openings,
    )


def extract_wall_segments(
    labels: np.ndarray,
    taxonomy: FloorplanTaxonomy,
    *,
    include_openings: bool = True,
    wall_policy: WallSkeletonPolicy | None = None,
) -> list[ReadoutWall]:
    """Extract wall segments independently from opening components."""

    labels = np.asarray(labels, dtype=np.int64)
    wall_mask = labels == taxonomy.class_id("wall")
    if include_openings:
        wall_mask |= np.isin(labels, np.asarray(taxonomy.opening_ids))
    walls, _skeleton = _skeleton_segments(
        wall_mask,
        policy=wall_policy,
    )
    return walls


__all__ = [
    "RasterWallPlan",
    "ReadoutFurniture",
    "ReadoutOpening",
    "ReadoutWall",
    "WallSkeletonPolicy",
    "extract_wall_segments",
    "raster_labels_to_wall_plan",
]
