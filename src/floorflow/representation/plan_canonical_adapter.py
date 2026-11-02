"""Lossless adaptation from RasterWallPlan to canonical CAD and tokens."""

from __future__ import annotations

import hashlib

import numpy as np
from scipy.ndimage import label as component_label
from shapely.geometry import Polygon

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.canonical_floorplan import (
    CanonicalFloorplan,
    CanonicalJunction,
    CanonicalOpening,
    CanonicalOperation,
    CanonicalRoom,
    CanonicalWall,
    canonical_fingerprint,
)
from floorflow.geometry.partition_polygons import component_cell_polygon
from floorflow.objectives.token_grammar import grammar_prefix_violations
from floorflow.representation.unified_floorplan_tokens import (
    CoordinateQuantizer,
    TokenizedFloorplan,
    canonicalize_floorplan,
    decode_floorplan,
    tokenize_floorplan,
)
from floorflow.representation.vector_floorplan_assembly import (
    adjacency_from_room_polygons,
)


def footprint_boundary_from_mask(
    footprint: np.ndarray,
) -> tuple[tuple[float, float], ...]:
    """Extract the largest exterior contour in canonical x/y order."""

    mask = np.asarray(footprint, dtype=bool)
    components, count = component_label(
        mask,
        structure=np.asarray(
            [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
            dtype=np.uint8,
        ),
    )
    if count == 0:
        raise ValueError("footprint mask has no exterior contour")
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    largest = components == int(np.argmax(sizes))
    polygon = component_cell_polygon(largest)
    coordinates = np.asarray(polygon.exterior.coords[:-1], dtype=np.float64)
    if len(coordinates) < 3:
        raise ValueError("footprint contour collapses during simplification")
    return tuple((float(x), float(y)) for x, y in coordinates)


def adaptive_coordinate_quantizer(
    size: tuple[int, int],
    *,
    reference_scale: int = 16,
    reference_resolution: int = 384,
) -> CoordinateQuantizer:
    scale = max(
        4,
        round(
            float(reference_scale)
            * min(size)
            / max(1, int(reference_resolution))
        ),
    )
    return CoordinateQuantizer(scale=scale)


def _opening_label(kind: str, taxonomy: FloorplanTaxonomy) -> str:
    if taxonomy.is_opening(kind):
        return kind
    prefix = "window:" if str(kind).startswith("window") else "door:"
    return next(
        (
            taxonomy.class_names[class_id]
            for class_id in taxonomy.opening_ids
            if taxonomy.class_names[class_id].startswith(prefix)
        ),
        taxonomy.class_names[taxonomy.opening_ids[0]],
    )


def raster_plan_to_canonical(
    plan,
    footprint: np.ndarray,
    taxonomy: FloorplanTaxonomy,
    *,
    case_id: str,
    track: str,
    source_hash: str | None = None,
    quantizer: CoordinateQuantizer | None = None,
) -> tuple[CanonicalFloorplan, float]:
    """Preserve explicit rooms, walls, openings, adjacency, and junctions."""

    boundary = footprint_boundary_from_mask(footprint)
    canonical_rooms = []
    for index, room in enumerate(plan.rooms, start=1):
        canonical_rooms.append(
            CanonicalRoom(
                id=str(room.get("id") or f"room_{index:04d}"),
                label=str(room["label"]),
                polygon=tuple(
                    tuple(map(float, point)) for point in room["polygon"]
                ),
                support=(
                    tuple(
                        tuple(map(float, point))
                        for point in room.get("support", ())
                    )
                    or None
                ),
                holes=tuple(
                    tuple(tuple(map(float, point)) for point in hole)
                    for hole in room.get("holes", ())
                ),
                confidence=float(room.get("confidence", 1.0)),
            )
        )
    canonical_walls = tuple(
        CanonicalWall(
            id=f"wall_{index:04d}",
            start=tuple(map(float, wall.start)),
            end=tuple(map(float, wall.end)),
            thickness=float(wall.thickness),
            wall_type=str(wall.wall_type),
        )
        for index, wall in enumerate(plan.walls, start=1)
    )
    openings = []
    dropped_openings = 0
    for index, opening in enumerate(plan.openings, start=1):
        if not 0 <= int(opening.wall_index) < len(canonical_walls):
            dropped_openings += 1
            continue
        openings.append(
            CanonicalOpening(
                id=f"opening_{index:04d}",
                label=_opening_label(opening.kind, taxonomy),
                center=tuple(map(float, opening.center)),
                width=float(opening.width),
                wall_id=canonical_walls[int(opening.wall_index)].id,
                confidence=1.0,
            )
        )
    source_junctions = getattr(plan, "junctions", ())
    junctions = tuple(
        CanonicalJunction(
            id=str(junction["id"]),
            point=tuple(map(float, junction["point"])),
            degree=int(junction["degree"]),
        )
        for junction in source_junctions
    )
    owner_cell_complex = getattr(plan, "owner_cell_complex", None)
    adjacency = (
        owner_cell_complex.adjacency
        if owner_cell_complex is not None
        else adjacency_from_room_polygons(
            [room.id for room in canonical_rooms],
            [
                Polygon(
                    room.polygon,
                    room.holes,
                )
                for room in canonical_rooms
            ],
        )
    )
    if source_hash is None:
        source_hash = hashlib.sha256(
            repr(
                (
                    plan.frame,
                    plan.rooms,
                    plan.walls,
                    plan.openings,
                )
            ).encode("utf-8")
        ).hexdigest()
    operations = (
        CanonicalOperation(
            "drop_detached_opening",
            "all",
            str(dropped_openings),
        ),
    ) if dropped_openings else ()
    plan_value = CanonicalFloorplan(
        case_id=str(case_id),
        track=str(track),
        width=int(footprint.shape[-1]),
        height=int(footprint.shape[0]),
        footprint=boundary,
        rooms=tuple(canonical_rooms),
        walls=canonical_walls,
        openings=tuple(openings),
        furniture=(),
        adjacency=tuple(sorted(tuple(sorted(edge)) for edge in adjacency)),
        source_hash=str(source_hash),
        junctions=junctions,
        operations=operations,
    )
    quantizer = quantizer or adaptive_coordinate_quantizer(
        (plan_value.width, plan_value.height)
    )
    return canonicalize_floorplan(
        plan_value,
        taxonomy,
        quantizer=quantizer,
    )


def canonical_token_roundtrip(
    plan: CanonicalFloorplan,
    taxonomy: FloorplanTaxonomy,
    *,
    quantizer: CoordinateQuantizer | None = None,
) -> tuple[TokenizedFloorplan, dict[str, float | int | bool]]:
    tokenized = tokenize_floorplan(
        plan,
        taxonomy,
        quantizer=quantizer,
    )
    decoded = decode_floorplan(
        tokenized.text,
        taxonomy,
        quantizer=quantizer,
    )
    original_fingerprint = canonical_fingerprint(tokenized.plan)
    decoded_fingerprint = canonical_fingerprint(decoded)
    violations = grammar_prefix_violations(tokenized.tokens)
    junction_degrees_exact = sorted(
        (junction.point, junction.degree)
        for junction in tokenized.plan.junctions
    ) == sorted(
        (junction.point, junction.degree)
        for junction in decoded.junctions
    )
    return tokenized, {
        "canonical_token_count": tokenized.token_count,
        "canonical_token_byte_count": tokenized.byte_count,
        "canonical_quantization_exact": tokenized.exact_quantization,
        "canonical_max_quantization_error": (
            tokenized.max_quantization_error
        ),
        "canonical_fingerprint_exact": (
            original_fingerprint == decoded_fingerprint
        ),
        "canonical_room_count_exact": (
            len(tokenized.plan.rooms) == len(decoded.rooms)
        ),
        "canonical_wall_count_exact": (
            len(tokenized.plan.walls) == len(decoded.walls)
        ),
        "canonical_opening_count_exact": (
            len(tokenized.plan.openings) == len(decoded.openings)
        ),
        "canonical_adjacency_exact": (
            tokenized.plan.adjacency == decoded.adjacency
        ),
        "canonical_junction_degree_exact": junction_degrees_exact,
        "canonical_prefix_violation_count": len(violations),
        "canonical_first_violation_index": (
            int(violations[0]["index"]) if violations else -1
        ),
    }


__all__ = [
    "adaptive_coordinate_quantizer",
    "canonical_token_roundtrip",
    "footprint_boundary_from_mask",
    "raster_plan_to_canonical",
]
