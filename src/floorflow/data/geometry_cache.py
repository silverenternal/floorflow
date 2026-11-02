"""Versioned vector-geometry cache derived from CubiCasa SVG annotations."""

from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from floorflow.data.svg import (
    _direct_polygon_points,
    _floor_nodes,
    _furniture_boundary_points,
    _numbers,
    _transform_points,
    parse_transform,
    rasterize_cubicasa_svg,
)
from floorflow.data.taxonomy import (
    FURNITURE_RARE,
    ROOM_RARE,
    FloorplanTaxonomy,
)

CACHE_VERSION = "geometry-v1"


@dataclass(frozen=True)
class PackedPolygons:
    points: np.ndarray
    offsets: np.ndarray


@dataclass(frozen=True)
class WallGeometry:
    polygon: np.ndarray
    centerline: np.ndarray
    thickness: float
    wall_type: str


@dataclass(frozen=True)
class OpeningGeometry:
    polygon: np.ndarray
    center: tuple[float, float]
    width: float
    class_id: int
    source_name: str
    wall_index: int | None
    host_distance: float | None


@dataclass(frozen=True)
class FloorplanGeometry:
    target: np.ndarray
    footprint: np.ndarray
    room_support: np.ndarray
    occlusion_mask: np.ndarray
    rooms: PackedPolygons
    room_class_ids: np.ndarray
    room_source_names: np.ndarray
    walls: PackedPolygons
    wall_centerlines: np.ndarray
    wall_thickness: np.ndarray
    wall_types: np.ndarray
    openings: PackedPolygons
    opening_centers: np.ndarray
    opening_widths: np.ndarray
    opening_class_ids: np.ndarray
    opening_source_names: np.ndarray
    opening_wall_indices: np.ndarray
    opening_host_distances: np.ndarray
    furniture: PackedPolygons
    furniture_class_ids: np.ndarray
    furniture_source_names: np.ndarray


def _normalise_transform(svg_path: Path, image_size: int):
    root = ET.parse(svg_path).getroot()
    view_box = _numbers(root.get("viewBox") or "")
    if len(view_box) != 4:
        min_x, min_y = 0.0, 0.0
        width = float(root.get("width") or image_size)
        height = float(root.get("height") or image_size)
    else:
        min_x, min_y, width, height = view_box
    if width <= 0.0 or height <= 0.0:
        raise ValueError(f"invalid SVG extent: {width}x{height}")
    margin = max(3, image_size // 64)
    scale = max(1.0, image_size - 2 * margin) / max(width, height)
    canvas_w = max(1, round(width * scale)) + 2 * margin
    canvas_h = max(1, round(height * scale)) + 2 * margin
    matrix = np.asarray(
        [
            [scale, 0.0, margin - min_x * scale],
            [0.0, scale, margin - min_y * scale],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    pad_x = max(0, image_size - canvas_w)
    pad_y = max(0, image_size - canvas_h)
    shift = np.asarray([pad_x // 2, pad_y // 2], dtype=np.float64)
    return root, matrix, shift


def _class_name(tokens: list[str], taxonomy: FloorplanTaxonomy, kind: str) -> str:
    source = tokens[1] if len(tokens) > 1 else ""
    if kind == "room":
        return taxonomy.room_map.get(source, ROOM_RARE)
    if kind == "furniture":
        return taxonomy.furniture_map.get(source, FURNITURE_RARE)
    prefix = "door:" if kind == "door" else "window:"
    return f"{prefix}{source or kind.title()}"


def _prominent_axis(
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    center = points.mean(axis=0)
    covariance = np.cov((points - center).T)
    values, vectors = np.linalg.eigh(covariance)
    major = vectors[:, int(values.argmax())]
    minor = np.asarray([-major[1], major[0]])
    u = (points - center) @ major
    v = (points - center) @ minor
    return center, major, minor, u, v


def _wall_from_polygon(points: np.ndarray, wall_type: str) -> WallGeometry:
    center, major, _minor, u, v = _prominent_axis(points)
    midpoint = center + major * (u.min() + u.max()) / 2.0
    direction = major * (u.max() - u.min()) / 2.0
    return WallGeometry(
        polygon=points,
        centerline=np.stack([midpoint - direction, midpoint + direction]),
        thickness=float(max(1.0, v.max() - v.min())),
        wall_type=wall_type,
    )


def _opening_from_polygon(
    points: np.ndarray,
    class_id: int,
    source_name: str,
) -> OpeningGeometry:
    center, _major, _minor, u, _v = _prominent_axis(points)
    return OpeningGeometry(
        polygon=points,
        center=(float(center[0]), float(center[1])),
        width=float(max(1.0, u.max() - u.min())),
        class_id=int(class_id),
        source_name=source_name,
        wall_index=None,
        host_distance=None,
    )


def _point_segment_distance(
    point: tuple[float, float],
    start: np.ndarray,
    end: np.ndarray,
) -> float:
    delta = end - start
    denominator = float(delta @ delta)
    if denominator <= 1.0e-9:
        return math.dist(point, start)
    t = float(np.clip((np.asarray(point) - start) @ delta / denominator, 0.0, 1.0))
    return float(np.linalg.norm(np.asarray(point) - (start + t * delta)))


def _attach_openings(
    openings: list[OpeningGeometry],
    walls: list[WallGeometry],
) -> list[OpeningGeometry]:
    attached = []
    for opening in openings:
        nearest = None
        distance = math.inf
        for index, wall in enumerate(walls):
            candidate = _point_segment_distance(
                opening.center,
                wall.centerline[0],
                wall.centerline[1],
            )
            if candidate < distance:
                distance = candidate
                nearest = index
        attached.append(
            OpeningGeometry(
                polygon=opening.polygon,
                center=opening.center,
                width=opening.width,
                class_id=opening.class_id,
                source_name=opening.source_name,
                wall_index=nearest,
                host_distance=None if nearest is None else distance,
            )
        )
    return attached


def pack_polygons(polygons: list[np.ndarray]) -> PackedPolygons:
    offsets = [0]
    points = []
    for polygon in polygons:
        values = np.asarray(polygon, dtype=np.float32)
        points.append(values)
        offsets.append(offsets[-1] + len(values))
    return PackedPolygons(
        points=(
            np.concatenate(points, axis=0)
            if points
            else np.zeros((0, 2), dtype=np.float32)
        ),
        offsets=np.asarray(offsets, dtype=np.int32),
    )


def unpack_polygons(packed: PackedPolygons) -> list[np.ndarray]:
    return [
        packed.points[packed.offsets[index] : packed.offsets[index + 1]]
        for index in range(len(packed.offsets) - 1)
    ]


def _draw_support(
    polygons: list[np.ndarray],
    image_size: int,
) -> np.ndarray:
    image = Image.new("L", (image_size, image_size), 0)
    draw = ImageDraw.Draw(image)
    for polygon in polygons:
        if len(polygon) >= 3:
            draw.polygon([tuple(map(float, point)) for point in polygon], fill=1)
    return np.asarray(image, dtype=np.uint8)


def rasterize_room_polygons(
    packed: PackedPolygons,
    class_ids: np.ndarray,
    image_size: int,
) -> np.ndarray:
    image = Image.new("L", (image_size, image_size), 0)
    draw = ImageDraw.Draw(image)
    for polygon, class_id in zip(unpack_polygons(packed), class_ids):
        if len(polygon) >= 3:
            draw.polygon(
                [tuple(map(float, point)) for point in polygon],
                fill=int(class_id),
            )
    return np.asarray(image, dtype=np.uint8)


def audit_geometry_hop(
    geometry: FloorplanGeometry,
    taxonomy: FloorplanTaxonomy,
    *,
    tolerance: float = 0.02,
) -> dict[str, object]:
    """Audit one data hop: cached polygons → raster support → taxonomy."""

    target = np.asarray(geometry.target)
    footprint = np.asarray(geometry.footprint) > 0
    support = np.asarray(geometry.room_support) > 0
    checks: dict[str, object] = {
        "target_shape": list(target.shape),
        "footprint_shape": list(footprint.shape),
        "room_support_shape": list(support.shape),
        "shape_match": (
            target.shape == footprint.shape == support.shape
        ),
    }
    valid_ids = (
        set(taxonomy.room_ids)
        | set(taxonomy.opening_ids)
        | set(taxonomy.furniture_ids)
        | {
        taxonomy.class_id("wall"),
        taxonomy.class_id("structure"),
        }
    )
    room_ids = {
        int(value) for value in np.asarray(geometry.room_class_ids).tolist()
    }
    opening_ids = {
        int(value) for value in np.asarray(geometry.opening_class_ids).tolist()
    }
    furniture_ids = {
        int(value) for value in np.asarray(geometry.furniture_class_ids).tolist()
    }
    checks["unknown_room_ids"] = sorted(room_ids - valid_ids)
    checks["unknown_opening_ids"] = sorted(
        opening_ids - valid_ids - {0}
    )
    opening_array = np.asarray(geometry.opening_class_ids)
    checks["unclassified_opening_count"] = (
        int((opening_array == 0).sum()) if opening_array.size else 0
    )
    checks["opening_count"] = int(opening_array.size)
    checks["unclassified_opening_fraction"] = (
        float(checks["unclassified_opening_count"]) / int(opening_array.size)
        if opening_array.size
        else 0.0
    )
    checks["unknown_furniture_ids"] = sorted(furniture_ids - valid_ids)
    checks["taxonomy_consistent"] = not any(
        checks[key]
        for key in (
            "unknown_room_ids",
            "unknown_opening_ids",
            "unknown_furniture_ids",
        )
    )
    rasterized = rasterize_room_polygons(
        geometry.rooms,
        geometry.room_class_ids,
        int(target.shape[-1]),
    ) > 0
    union = int(np.logical_or(rasterized, support).sum())
    intersection = int(np.logical_and(rasterized, support).sum())
    support_iou = 1.0 if union == 0 else float(intersection / union)
    checks["room_support_iou"] = support_iou
    checks["room_support_within_tolerance"] = (
        support_iou >= 1.0 - float(tolerance)
    )
    outside = int(np.logical_and(rasterized, ~footprint).sum())
    checks["room_pixels_outside_footprint"] = outside
    checks["footprint_consistent"] = outside == 0
    checks["consistent"] = bool(
        checks["shape_match"]
        and checks["taxonomy_consistent"]
        and checks["room_support_within_tolerance"]
        and checks["footprint_consistent"]
    )
    return {"consistent": checks["consistent"], "checks": checks}


def build_floorplan_geometry(
    svg_path: str | Path,
    *,
    floor_id: str,
    image_size: int,
    taxonomy: FloorplanTaxonomy,
) -> FloorplanGeometry:
    svg_path = Path(svg_path)
    root, normalise, shift = _normalise_transform(svg_path, image_size)
    floors = _floor_nodes(root, normalise)
    lookup = {item_id: parent for item_id, parent in floors}
    requested = floor_id if floor_id.startswith("Floor-") else f"Floor-{floor_id}"
    if requested not in lookup:
        raise ValueError(f"floor {requested!r} not found; available={sorted(lookup)}")
    floor_node = next(
        node for node in root.iter() if (node.get("id") or "") == requested
    )
    room_polygons: list[np.ndarray] = []
    room_class_ids: list[int] = []
    room_source_names: list[str] = []
    wall_polygons: list[np.ndarray] = []
    walls: list[WallGeometry] = []
    openings: list[OpeningGeometry] = []
    furniture_polygons: list[np.ndarray] = []
    furniture_class_ids: list[int] = []
    furniture_source_names: list[str] = []

    def visit(element: ET.Element, parent_matrix: np.ndarray) -> None:
        matrix = parent_matrix @ parse_transform(element.get("transform"))
        element_id = element.get("id") or ""
        class_name = element.get("class") or ""
        tokens = class_name.split()
        is_room = "Space" in tokens and "Outdoor" not in tokens
        if is_room:
            target_name = _class_name(tokens, taxonomy, "room")
            if target_name not in taxonomy.class_names:
                target_name = ROOM_RARE
            for polygon in _direct_polygon_points(element, matrix):
                shifted = _transform_points(polygon, np.eye(3)) + shift
                room_polygons.append(shifted)
                room_class_ids.append(taxonomy.class_id(target_name))
                room_source_names.append(tokens[1] if len(tokens) > 1 else "Undefined")
        if element_id == "Wall":
            wall_type = "exterior" if "External" in tokens else "interior"
            for polygon in _direct_polygon_points(element, matrix):
                shifted = _transform_points(polygon, np.eye(3)) + shift
                wall_polygons.append(shifted)
                walls.append(_wall_from_polygon(shifted, wall_type))
        elif element_id in {"Door", "Window"}:
            kind = "door" if element_id == "Door" else "window"
            target_name = _class_name(tokens, taxonomy, kind)
            if target_name not in taxonomy.class_names:
                target_name = f"{kind}:{kind.title()}"
            for polygon in _direct_polygon_points(element, matrix):
                shifted = _transform_points(polygon, np.eye(3)) + shift
                openings.append(
                    _opening_from_polygon(
                        shifted,
                        taxonomy.class_id(target_name),
                        tokens[1] if len(tokens) > 1 else element_id,
                    )
                )
        elif "FixedFurniture" in tokens and "FixedFurnitureSet" not in tokens:
            target_name = _class_name(tokens, taxonomy, "furniture")
            if target_name not in taxonomy.class_names:
                target_name = FURNITURE_RARE
            for polygon in _furniture_boundary_points(element, matrix):
                shifted = _transform_points(polygon, np.eye(3)) + shift
                furniture_polygons.append(shifted)
                furniture_class_ids.append(taxonomy.class_id(target_name))
                furniture_source_names.append(
                    tokens[1] if len(tokens) > 1 else "Misc"
                )
        for child in element:
            visit(child, matrix)

    visit(floor_node, lookup[requested])
    target, footprint = rasterize_cubicasa_svg(
        svg_path,
        image_size=image_size,
        floor_id=requested,
        taxonomy=taxonomy,
    )
    room_support = _draw_support(room_polygons, image_size)
    visible_rooms = np.isin(target, np.asarray(taxonomy.room_ids))
    occlusion_mask = (
        (room_support > 0)
        & ~visible_rooms
        & (footprint > 0)
    ).astype(np.uint8)
    openings = _attach_openings(openings, walls)
    return FloorplanGeometry(
        target=target,
        footprint=footprint,
        room_support=room_support,
        occlusion_mask=occlusion_mask,
        rooms=pack_polygons(room_polygons),
        room_class_ids=np.asarray(room_class_ids, dtype=np.int32),
        room_source_names=np.asarray(room_source_names, dtype=np.str_),
        walls=pack_polygons(wall_polygons),
        wall_centerlines=np.asarray(
            [wall.centerline for wall in walls],
            dtype=np.float32,
        ).reshape(-1, 2, 2),
        wall_thickness=np.asarray(
            [wall.thickness for wall in walls],
            dtype=np.float32,
        ),
        wall_types=np.asarray(
            [wall.wall_type for wall in walls],
            dtype=np.str_,
        ),
        openings=pack_polygons([opening.polygon for opening in openings]),
        opening_centers=np.asarray(
            [opening.center for opening in openings],
            dtype=np.float32,
        ).reshape(-1, 2),
        opening_widths=np.asarray(
            [opening.width for opening in openings],
            dtype=np.float32,
        ),
        opening_class_ids=np.asarray(
            [opening.class_id for opening in openings],
            dtype=np.int32,
        ),
        opening_source_names=np.asarray(
            [opening.source_name for opening in openings],
            dtype=np.str_,
        ),
        opening_wall_indices=np.asarray(
            [
                -1 if opening.wall_index is None else opening.wall_index
                for opening in openings
            ],
            dtype=np.int32,
        ),
        opening_host_distances=np.asarray(
            [
                -1.0 if opening.host_distance is None else opening.host_distance
                for opening in openings
            ],
            dtype=np.float32,
        ),
        furniture=pack_polygons(furniture_polygons),
        furniture_class_ids=np.asarray(furniture_class_ids, dtype=np.int32),
        furniture_source_names=np.asarray(
            furniture_source_names,
            dtype=np.str_,
        ),
    )


def geometry_cache_path(
    cache_root: str | Path,
    split: str,
    relative_case: str | Path,
    floor_id: str,
) -> Path:
    filename = f"{floor_id.replace('-', '_')}.geometry.npz"
    return Path(cache_root) / split / Path(relative_case) / filename


def write_geometry_cache(
    geometry: FloorplanGeometry,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> bool:
    path = Path(path)
    if path.is_file() and not overwrite:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".tmp-{os.getpid()}.npz")
    np.savez_compressed(
        temporary,
        cache_version=np.asarray(CACHE_VERSION),
        target=geometry.target,
        footprint=geometry.footprint,
        room_support=geometry.room_support,
        occlusion_mask=geometry.occlusion_mask,
        room_points=geometry.rooms.points,
        room_offsets=geometry.rooms.offsets,
        room_class_ids=geometry.room_class_ids,
        room_source_names=geometry.room_source_names,
        wall_points=geometry.walls.points,
        wall_offsets=geometry.walls.offsets,
        wall_centerlines=geometry.wall_centerlines,
        wall_thickness=geometry.wall_thickness,
        wall_types=geometry.wall_types,
        opening_points=geometry.openings.points,
        opening_offsets=geometry.openings.offsets,
        opening_centers=geometry.opening_centers,
        opening_widths=geometry.opening_widths,
        opening_class_ids=geometry.opening_class_ids,
        opening_source_names=geometry.opening_source_names,
        opening_wall_indices=geometry.opening_wall_indices,
        opening_host_distances=geometry.opening_host_distances,
        furniture_points=geometry.furniture.points,
        furniture_offsets=geometry.furniture.offsets,
        furniture_class_ids=geometry.furniture_class_ids,
        furniture_source_names=geometry.furniture_source_names,
    )
    os.replace(temporary, path)
    return True


def load_geometry_cache(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        if str(payload["cache_version"]) != CACHE_VERSION:
            raise ValueError(f"unsupported geometry cache version: {payload['cache_version']}")
        return {key: payload[key] for key in payload.files}


def geometry_from_payload(payload: dict[str, np.ndarray]) -> FloorplanGeometry:
    return FloorplanGeometry(
        target=payload["target"],
        footprint=payload["footprint"],
        room_support=payload["room_support"],
        occlusion_mask=payload["occlusion_mask"],
        rooms=PackedPolygons(
            payload["room_points"],
            payload["room_offsets"],
        ),
        room_class_ids=payload["room_class_ids"],
        room_source_names=payload["room_source_names"],
        walls=PackedPolygons(
            payload["wall_points"],
            payload["wall_offsets"],
        ),
        wall_centerlines=payload["wall_centerlines"],
        wall_thickness=payload["wall_thickness"],
        wall_types=payload["wall_types"],
        openings=PackedPolygons(
            payload["opening_points"],
            payload["opening_offsets"],
        ),
        opening_centers=payload["opening_centers"],
        opening_widths=payload["opening_widths"],
        opening_class_ids=payload["opening_class_ids"],
        opening_source_names=payload["opening_source_names"],
        opening_wall_indices=payload["opening_wall_indices"],
        opening_host_distances=payload["opening_host_distances"],
        furniture=PackedPolygons(
            payload["furniture_points"],
            payload["furniture_offsets"],
        ),
        furniture_class_ids=payload["furniture_class_ids"],
        furniture_source_names=payload["furniture_source_names"],
    )
