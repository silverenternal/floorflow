"""Deterministic vector CAD assembly shared by RPLAN targets and generation."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from shapely.geometry import LineString, Point, Polygon

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.canonical_floorplan import (
    CanonicalFloorplan,
    CanonicalOpening,
    CanonicalOperation,
    CanonicalRoom,
    CanonicalWall,
)
from floorflow.geometry.cad_wall_candidates import room_boundary_candidates
from floorflow.geometry.learned_wall_fusion import fuse_learned_wall_endpoints
from floorflow.geometry.owner_cell_complex import OwnerCellComplex
from floorflow.representation.canonical_assembly_support import (
    adaptive_assembly_quantizer,
    canonical_junctions_from_rows,
    canonical_openings_from_rows,
)
from floorflow.representation.unified_floorplan_tokens import (
    CoordinateQuantizer,
    canonicalize_floorplan,
)

_WALL_CLOSURE_TOLERANCE = 2.0
_MAX_WALL_CLOSURE_GAP = 30.0


def _wall_key(
    start: tuple[float, float],
    end: tuple[float, float],
) -> tuple[tuple[float, float], tuple[float, float]]:
    rounded = (
        (round(start[0], 3), round(start[1], 3)),
        (round(end[0], 3), round(end[1], 3)),
    )
    return tuple(sorted(rounded))


def _canonical_walls_from_candidates(
    candidates: Sequence,
    footprint: Polygon,
) -> tuple[CanonicalWall, ...]:
    unique = {}
    for wall in candidates:
        key = _wall_key(wall.start, wall.end)
        midpoint = Point(
            0.5 * (wall.start[0] + wall.end[0]),
            0.5 * (wall.start[1] + wall.end[1]),
        )
        unique[key] = CanonicalWall(
            id="",
            start=key[0],
            end=key[1],
            thickness=float(wall.thickness),
            wall_type=(
                "exterior"
                if midpoint.distance(footprint.boundary) <= 1.0
                else "interior"
            ),
        )
    return tuple(
        CanonicalWall(
            id=f"wall_{index:04d}",
            start=wall.start,
            end=wall.end,
            thickness=wall.thickness,
            wall_type=wall.wall_type,
        )
        for index, wall in enumerate(unique.values(), start=1)
    )


def _close_open_wall_gaps(
    walls: Sequence[CanonicalWall],
    footprint: Polygon,
    *,
    cluster_tolerance: float = _WALL_CLOSURE_TOLERANCE,
    maximum_gap: float = _MAX_WALL_CLOSURE_GAP,
    minimum_connector_length: float = 2.01,
) -> tuple[CanonicalWall, ...]:
    """Connect paired open endpoints introduced by bounded boundary splits."""

    if not walls:
        return ()
    endpoints = [
        (wall.id, endpoint_index, point)
        for wall in walls
        for endpoint_index, point in enumerate((wall.start, wall.end))
    ]
    degrees = []
    for _wall_id, _endpoint_index, point in endpoints:
        degrees.append(
            sum(
                math.dist(point, other) <= cluster_tolerance
                for _other_wall, _other_endpoint, other in endpoints
            )
        )
    open_endpoints = [
        endpoint
        for endpoint, degree in zip(endpoints, degrees, strict=True)
        if degree < 2
    ]
    used: set[tuple[str, int]] = set()
    connector_indices = [
        int(wall.id.split("_", 1)[1])
        for wall in walls
        if wall.id.startswith("connector_")
        and wall.id.split("_", 1)[1].isdigit()
    ]
    next_connector = max(connector_indices, default=0) + 1
    connectors = []
    for wall_id, endpoint_index, point in open_endpoints:
        key = (wall_id, endpoint_index)
        if key in used:
            continue
        candidates = [
            (math.dist(point, other), other_wall, other_endpoint, other)
            for other_wall, other_endpoint, other in open_endpoints
            if (
                (other_wall, other_endpoint) not in used
                and other_wall != wall_id
                and max(cluster_tolerance, minimum_connector_length)
                < math.dist(point, other)
                <= maximum_gap
            )
        ]
        if not candidates:
            candidates = [
                (math.dist(point, other), other_wall, other_endpoint, other)
                for other_wall, other_endpoint, other in endpoints
                if (
                    (other_wall, other_endpoint) not in used
                    and other_wall != wall_id
                    and max(
                        cluster_tolerance,
                        minimum_connector_length,
                    )
                    < math.dist(point, other)
                    <= maximum_gap
                )
            ]
        if not candidates:
            continue
        _distance, other_wall, other_endpoint, other = min(candidates)
        used.add(key)
        used.add((other_wall, other_endpoint))
        midpoint = (0.5 * (point[0] + other[0]), 0.5 * (point[1] + other[1]))
        connectors.append(
            CanonicalWall(
                id=f"connector_{next_connector + len(connectors):04d}",
                start=point,
                end=other,
                thickness=3.0,
                wall_type=(
                    "exterior"
                    if Point(midpoint).distance(footprint.boundary) <= 1.5
                    else "interior"
                ),
            )
        )
    if not connectors:
        return tuple(walls)
    return (*walls, *connectors)


def walls_from_room_polygons(
    room_rows: list[dict],
    footprint: Polygon,
) -> tuple[CanonicalWall, ...]:
    """Vectorize unique room boundaries and classify footprint exposure."""

    ring_rows = [
        *room_rows,
        {"polygon": list(footprint.exterior.coords[:-1]), "holes": []},
    ]
    return _canonical_walls_from_candidates(
        room_boundary_candidates(ring_rows),
        footprint,
    )


def entrance_opening_from_boundary(
    boundary: tuple[tuple[float, float], ...],
    walls: tuple[CanonicalWall, ...],
    *,
    interval: Sequence[Sequence[float]] | None = None,
    label: str = "door:Door",
) -> CanonicalOpening | None:
    """Use the supplied entrance interval, or the first boundary segment."""

    if not walls:
        return None
    if interval is None:
        if len(boundary) < 2:
            return None
        start, end = boundary[0], boundary[1]
    else:
        if len(interval) != 2:
            raise ValueError("entrance interval must contain two endpoints")
        start = tuple(map(float, interval[0]))
        end = tuple(map(float, interval[1]))
    width = math.dist(start, end)
    if width <= 0.0:
        return None
    center = (0.5 * (start[0] + end[0]), 0.5 * (start[1] + end[1]))
    center_point = Point(center)
    exterior = [
        wall
        for wall in walls
        if wall.wall_type == "exterior"
        and center_point.distance(LineString([wall.start, wall.end])) <= 1.0
    ]
    host = min(
        exterior or list(walls),
        key=lambda wall: center_point.distance(
            LineString([wall.start, wall.end])
        ),
    )
    return CanonicalOpening(
        id="opening_0001",
        label=label,
        center=center,
        width=width,
        wall_id=host.id,
    )


def adjacency_from_room_polygons(
    room_ids: Sequence[str],
    polygons: Sequence[Polygon],
    *,
    tolerance: float = 1.5,
) -> tuple[tuple[str, str], ...]:
    edges = []
    for left in range(len(polygons)):
        for right in range(left + 1, len(polygons)):
            contact = polygons[left].intersection(polygons[right])
            if (
                polygons[left].distance(polygons[right]) <= tolerance
                and contact.length > 1.0e-6
            ):
                edges.append(tuple(sorted((room_ids[left], room_ids[right]))))
    return tuple(sorted(set(edges)))


def assemble_vector_floorplan(
    *,
    case_id: str,
    track: str,
    width: int,
    height: int,
    footprint: Sequence[Sequence[float]],
    rooms: Sequence[dict],
    source_hash: str,
    taxonomy: FloorplanTaxonomy,
    operations: tuple[CanonicalOperation, ...] = (),
    adjacency: Sequence[Sequence[str]] | None = None,
    owner_cell_complex: OwnerCellComplex | None = None,
    wall_source: str | None = None,
    wall_candidates_override: Sequence[CanonicalWall] | None = None,
    openings: Sequence[dict] | None = None,
    junctions: Sequence[dict] | None = None,
    learned_wall_endpoints: Sequence[Sequence[Sequence[float]]] | None = None,
    learned_wall_scores: Sequence[float] | None = None,
    learned_wall_fusion_weight: float = 0.0,
    include_entrance: bool = True,
    entrance_interval: Sequence[Sequence[float]] | None = None,
    quantizer: CoordinateQuantizer | None = None,
    adaptive_quantizer: bool = True,
    max_wall_gap_fraction: float = 0.08,
) -> CanonicalFloorplan:
    """Assemble one validated vector plan from explicit room polygons."""

    boundary = tuple(tuple(map(float, point)) for point in footprint)
    footprint_shape = Polygon(boundary)
    if not footprint_shape.is_valid:
        raise ValueError("footprint is invalid")
    canonical_rooms = []
    room_rows = []
    polygons = []
    for index, room in enumerate(rooms, start=1):
        room_id = str(room.get("id") or f"room_{index:02d}")
        label = str(room["label"])
        if not taxonomy.is_room(label):
            raise ValueError(f"illegal room label: {label}")
        ring = tuple(tuple(map(float, point)) for point in room["polygon"])
        shape = Polygon(ring, room.get("holes", ()))
        if not shape.is_valid or shape.area <= 0.0:
            raise ValueError(f"room {room_id} has invalid geometry")
        canonical_rooms.append(
            CanonicalRoom(
                room_id,
                label,
                ring,
                support=ring,
                holes=tuple(
                    tuple(tuple(map(float, point)) for point in hole)
                    for hole in room.get("holes", ())
                ),
                confidence=float(room.get("confidence", 1.0)),
            )
        )
        room_rows.append(
            {
                "label": label,
                "polygon": ring,
                "holes": room.get("holes", ()),
            }
        )
        polygons.append(shape)
    if not canonical_rooms:
        raise ValueError("floorplan has no rooms")
    assembly_operations = list(operations)
    maximum_gap = max(
        2.0,
        float(max_wall_gap_fraction) * min(int(width), int(height)),
    )
    wall_source = wall_source or (
        "owner_cells" if owner_cell_complex is not None else "room_polygons"
    )
    if wall_candidates_override is not None:
        unclosed_count = len(wall_candidates_override)
        walls = list(
            _close_open_wall_gaps(
                wall_candidates_override,
                footprint_shape,
                maximum_gap=maximum_gap,
            )
        )
        assembly_operations.append(
            CanonicalOperation(
                "owner_action_wall_selection",
                "all",
                (
                    f"walls={len(walls)};"
                    f"connectors={len(walls) - unclosed_count}"
                ),
            )
        )
    elif wall_source not in {
        "room_polygons",
        "shared_owners",
        "shared_network",
        "shared_graph",
        "owner_cells",
    }:
        raise ValueError(
            "wall_source must be room_polygons, shared_owners, "
            "shared_network, shared_graph, or owner_cells"
        )
    if (
        wall_source
        in {
            "shared_owners",
            "shared_network",
            "shared_graph",
            "owner_cells",
        }
        and owner_cell_complex is None
    ):
        raise ValueError(f"{wall_source} requires an owner cell complex")
    if wall_candidates_override is not None:
        pass
    elif wall_source == "shared_graph":
        walls = list(
            _canonical_walls_from_candidates(
                owner_cell_complex.wall_candidates(
                    minimum_length=2.01,
                    include_void=True,
                ),
                footprint_shape,
            )
        )
        unclosed_count = len(walls)
        walls = list(
            _close_open_wall_gaps(
                walls,
                footprint_shape,
                maximum_gap=maximum_gap,
            )
        )
        assembly_operations.append(
            CanonicalOperation(
                "shared_graph_wall_projection",
                "all",
                (
                    f"boundaries={len(walls)};"
                    f"connectors={len(walls) - unclosed_count}"
                ),
            )
        )
    elif wall_source == "shared_network":
        walls = list(
            _canonical_walls_from_candidates(
                owner_cell_complex.merged_wall_candidates(),
                footprint_shape,
            )
        )
        unclosed_count = len(walls)
        walls = list(
            _close_open_wall_gaps(
                walls,
                footprint_shape,
                maximum_gap=maximum_gap,
            )
        )
        assembly_operations.append(
            CanonicalOperation(
                "shared_network_wall_projection",
                "all",
                (
                    f"boundaries={len(walls)};"
                    f"connectors={len(walls) - unclosed_count}"
                ),
            )
        )
    elif wall_source == "shared_owners":
        interior_candidates = list(
            owner_cell_complex.wall_candidates(
                minimum_length=2.01,
                include_void=False,
            )
        )
        split_points = [
            point
            for wall in interior_candidates
            for point in (wall.start, wall.end)
        ]
        exterior_candidates = room_boundary_candidates(
            [
                {
                    "polygon": list(footprint_shape.exterior.coords[:-1]),
                    "holes": [],
                }
            ],
            split_points=split_points,
            minimum_length=2.01,
        )
        walls = list(
            _canonical_walls_from_candidates(
                [*interior_candidates, *exterior_candidates],
                footprint_shape,
            )
        )
        unclosed_count = len(walls)
        walls = list(
            _close_open_wall_gaps(
                walls,
                footprint_shape,
                maximum_gap=maximum_gap,
            )
        )
        assembly_operations.append(
            CanonicalOperation(
                "shared_owner_wall_projection",
                "all",
                (
                    f"boundaries={len(walls)};"
                    f"connectors={len(walls) - unclosed_count}"
                ),
            )
        )
    elif wall_source == "owner_cells":
        walls = list(
            _canonical_walls_from_candidates(
                owner_cell_complex.wall_candidates(
                    minimum_length=1.0,
                    include_void=True,
                ),
                footprint_shape,
            )
        )
        assembly_operations.append(
            CanonicalOperation(
                "owner_cell_wall_projection",
                "all",
                f"boundaries={len(walls)}",
            )
        )
    else:
        walls = list(walls_from_room_polygons(room_rows, footprint_shape))
    edited_clusters = 0
    if (
        learned_wall_endpoints is not None
        and learned_wall_scores is not None
        and float(learned_wall_fusion_weight) > 0.0
    ):
        fused, edited_clusters = fuse_learned_wall_endpoints(
            walls,
            np.asarray(learned_wall_endpoints, dtype=np.float64),
            np.asarray(learned_wall_scores, dtype=np.float64),
            footprint_shape,
            weight=float(learned_wall_fusion_weight),
        )
        walls = list(fused)
        assembly_operations.append(
            CanonicalOperation(
                "learned_wall_endpoint_fusion",
                "all",
                f"clusters={edited_clusters}",
            )
        )
    provided_openings = canonical_openings_from_rows(
        openings,
        [wall.id for wall in walls],
    )
    include_entrance = include_entrance and not provided_openings
    opening = (
        entrance_opening_from_boundary(
            boundary,
            walls,
            interval=entrance_interval,
        )
        if include_entrance
        else None
    )
    if opening is not None and opening.wall_id in {wall.id for wall in walls}:
        host = next(wall for wall in walls if wall.id == opening.wall_id)
        host_distance = Point(opening.center).distance(
            LineString([host.start, host.end])
        )
        if host_distance > 0.25 and entrance_interval is not None:
            host = CanonicalWall(
                id=f"wall_{len(walls) + 1:04d}",
                start=tuple(map(float, entrance_interval[0])),
                end=tuple(map(float, entrance_interval[1])),
                thickness=3.0,
                wall_type="exterior",
            )
            walls.append(host)
            opening = CanonicalOpening(
                id=opening.id,
                label=opening.label,
                center=opening.center,
                width=opening.width,
                wall_id=host.id,
                confidence=opening.confidence,
            )
            if wall_source == "shared_graph":
                walls = list(
                    _close_open_wall_gaps(
                        walls,
                        footprint_shape,
                        maximum_gap=maximum_gap,
                    )
                )
    plan = CanonicalFloorplan(
        case_id=case_id,
        track=track,
        width=int(width),
        height=int(height),
        footprint=boundary,
        rooms=tuple(canonical_rooms),
        walls=tuple(walls),
        openings=(
            provided_openings
            if provided_openings
            else (opening,) if opening is not None else ()
        ),
        furniture=(),
        adjacency=(
            tuple(tuple(sorted(map(str, edge))) for edge in adjacency)
            if adjacency is not None
            else owner_cell_complex.adjacency
            if owner_cell_complex is not None
            else adjacency_from_room_polygons(
                [room.id for room in canonical_rooms],
                polygons,
            )
        ),
        source_hash=source_hash,
        junctions=canonical_junctions_from_rows(junctions),
        operations=tuple(assembly_operations),
    )
    if quantizer is None and adaptive_quantizer:
        quantizer = adaptive_assembly_quantizer(width, height)
    canonical, _ = canonicalize_floorplan(
        plan,
        taxonomy,
        quantizer=quantizer or CoordinateQuantizer(scale=16),
    )
    return canonical


__all__ = [
    "adjacency_from_room_polygons",
    "assemble_vector_floorplan",
    "entrance_opening_from_boundary",
    "walls_from_room_polygons",
]
