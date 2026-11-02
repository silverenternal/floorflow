"""Reversible vector-first floorplan grammar shared by every CAD track."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, replace
from itertools import pairwise

from shapely.geometry import LineString, Point, Polygon

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.canonical_floorplan import (
    CanonicalFloorplan,
    CanonicalWall,
    validate_floorplan,
)
from floorflow.evaluation.fevp_metric_contracts import FEVPTolerances
from floorflow.representation.floorplan_token_io import (
    TOKEN_SCHEMA_VERSION,
    CoordinateQuantizer,
    FloorplanToken,
    parse_tokens,
    serialize_tokens,
)
from floorflow.representation.floorplan_token_io import atom as _atom
from floorflow.representation.junction_canonicalization import derive_junctions


@dataclass(frozen=True)
class TokenizedFloorplan:
    plan: CanonicalFloorplan
    tokens: tuple[FloorplanToken, ...]
    exact_quantization: bool
    max_quantization_error: float

    @property
    def text(self) -> str:
        return serialize_tokens(self.tokens)

    @property
    def token_count(self) -> int:
        return len(self.tokens)

    @property
    def byte_count(self) -> int:
        return len(self.text.encode("utf-8"))


def _coordinates(points: Iterable[Iterable[float]], name: str) -> list[tuple[float, float]]:
    output = []
    for point in points:
        values = tuple(float(value) for value in point)
        if len(values) != 2 or not all(math.isfinite(value) for value in values):
            raise ValueError(f"{name} contains a non-finite or non-2D point")
        output.append((values[0], values[1]))
    if len(output) >= 2 and output[0] == output[-1]:
        output.pop()
    return output


def _canonical_ring(
    points: Iterable[Iterable[float]], quantizer: CoordinateQuantizer, *,
    name: str, clockwise: bool,
) -> tuple[tuple[int, int], ...]:
    values = [
        (quantizer.quantize(x), quantizer.quantize(y))
        for x, y in _coordinates(points, name)
    ]
    deduplicated = [
        point for index, point in enumerate(values) if index == 0 or point != values[index - 1]
    ]
    if len(deduplicated) >= 2 and deduplicated[0] == deduplicated[-1]:
        deduplicated.pop()
    if len(deduplicated) < 3:
        raise ValueError(f"{name} needs at least three distinct vertices")
    area_twice = sum(
        left[0] * right[1] - right[0] * left[1]
        for left, right in zip(
            deduplicated,
            deduplicated[1:] + deduplicated[:1],
            strict=True,
        )
    )
    if area_twice == 0:
        raise ValueError(f"{name} has zero area")
    if (area_twice > 0) == clockwise:
        deduplicated = list(reversed(deduplicated))
    start = min(range(len(deduplicated)), key=deduplicated.__getitem__)
    return tuple(deduplicated[start:] + deduplicated[:start])


def _ring_float(
    ring: tuple[tuple[int, int], ...],
    quantizer: CoordinateQuantizer,
) -> tuple[tuple[float, float], ...]:
    return tuple(
        (quantizer.dequantize(x), quantizer.dequantize(y))
        for x, y in ring
    )


def _canonical_point(point: tuple[float, float], quantizer: CoordinateQuantizer) -> tuple[float, float]:
    return (
        quantizer.dequantize(quantizer.quantize(point[0])),
        quantizer.dequantize(quantizer.quantize(point[1])),
    )


def _q(value: float, quantizer: CoordinateQuantizer) -> float:
    return quantizer.dequantize(quantizer.quantize(value))


def _canonical_wall(wall: CanonicalWall, quantizer: CoordinateQuantizer) -> CanonicalWall:
    start, end = sorted(
        (
            _canonical_point(wall.start, quantizer),
            _canonical_point(wall.end, quantizer),
        )
    )
    return replace(
        wall,
        start=start,
        end=end,
        thickness=_q(wall.thickness, quantizer),
    )


def _quantization_error(plan: CanonicalFloorplan, quantizer: CoordinateQuantizer) -> float:
    values: list[float] = []
    rings = [plan.footprint]
    rings.extend(room.polygon for room in plan.rooms)
    rings.extend(room.support or () for room in plan.rooms)
    rings.extend(hole for room in plan.rooms for hole in room.holes)
    rings.extend(item.polygon for item in plan.furniture)
    values.extend(value for ring in rings for point in ring for value in point)
    values.extend(value for wall in plan.walls for point in (wall.start, wall.end) for value in point)
    values.extend(wall.thickness for wall in plan.walls)
    values.extend(value for opening in plan.openings for value in opening.center)
    values.extend(opening.width for opening in plan.openings)
    values.extend(room.confidence for room in plan.rooms)
    values.extend(opening.confidence for opening in plan.openings)
    values.extend(item.confidence for item in plan.furniture)
    values.extend(value for junction in plan.junctions for value in junction.point)
    if not values:
        return 0.0
    return max(abs(value - quantizer.dequantize(quantizer.quantize(value))) for value in values)


def canonicalize_floorplan(
    plan: CanonicalFloorplan, taxonomy: FloorplanTaxonomy, *,
    quantizer: CoordinateQuantizer | None = None,
    tolerances: FEVPTolerances | None = None,
) -> tuple[CanonicalFloorplan, float]:
    """Canonicalize ordering, winding, and fixed-point coordinates."""

    quantizer = quantizer or CoordinateQuantizer()
    tolerances = tolerances or FEVPTolerances()
    max_error = _quantization_error(plan, quantizer)
    footprint = _ring_float(
        _canonical_ring(plan.footprint, quantizer, name="footprint", clockwise=False),
        quantizer,
    )
    rooms = []
    for room in plan.rooms:
        holes = tuple(
            _ring_float(
                _canonical_ring(hole, quantizer, name=f"room {room.id} hole", clockwise=True),
                quantizer,
            )
            for hole in room.holes
        )
        rooms.append(
            replace(
                room,
                polygon=_ring_float(
                    _canonical_ring(
                        room.polygon,
                        quantizer,
                        name=f"room {room.id}",
                        clockwise=False,
                    ),
                    quantizer,
                ),
                support=(
                    _ring_float(
                        _canonical_ring(
                            room.support,
                            quantizer,
                            name=f"room {room.id} support",
                            clockwise=False,
                        ),
                        quantizer,
                    )
                    if room.support
                    else None
                ),
                holes=tuple(sorted(holes)),
                confidence=_q(room.confidence, quantizer),
            )
        )
    walls = tuple(
        sorted(
            (_canonical_wall(wall, quantizer) for wall in plan.walls),
            key=lambda wall: wall.id,
        )
    )
    rooms.sort(key=lambda room: room.id)
    adjacency = tuple(sorted(tuple(sorted(edge)) for edge in plan.adjacency))
    if len(set(adjacency)) != len(adjacency):
        raise ValueError("duplicate adjacency edge")
    openings = tuple(
        sorted(
            (
                replace(
                    opening,
                    center=(
                        quantizer.dequantize(quantizer.quantize(opening.center[0])),
                        quantizer.dequantize(quantizer.quantize(opening.center[1])),
                    ),
                    width=quantizer.dequantize(quantizer.quantize(opening.width)),
                    confidence=_q(opening.confidence, quantizer),
                )
                for opening in plan.openings
            ),
            key=lambda opening: opening.id,
        )
    )
    furniture = tuple(
        sorted(
            (
                replace(
                    item,
                    polygon=_ring_float(
                        _canonical_ring(
                            item.polygon,
                            quantizer,
                            name=f"furniture {item.id}",
                            clockwise=False,
                        ),
                        quantizer,
                    ),
                    confidence=_q(item.confidence, quantizer),
                )
                for item in plan.furniture
            ),
            key=lambda item: item.id,
        )
    )
    junctions = (
        derive_junctions(
            walls,
            quantizer,
            tolerances.wall_junction_cluster_px,
        )
        if not plan.junctions
        else tuple(
            sorted(
                (
                    replace(
                        junction,
                        point=(
                            quantizer.dequantize(quantizer.quantize(junction.point[0])),
                            quantizer.dequantize(quantizer.quantize(junction.point[1])),
                        ),
                    )
                    for junction in plan.junctions
                ),
                key=lambda junction: (junction.point, junction.id),
            )
        )
    )
    operations = tuple(
        sorted(plan.operations, key=lambda item: (item.kind, item.target_id, item.detail))
    )
    canonical = replace(
        plan,
        footprint=footprint,
        rooms=tuple(rooms),
        walls=walls,
        openings=openings,
        furniture=furniture,
        adjacency=adjacency,
        junctions=junctions,
        operations=operations,
    )
    errors = validate_vector_floorplan(
        canonical,
        taxonomy,
        tolerances=tolerances,
        quantizer=quantizer,
    )
    if errors:
        raise ValueError("\n".join(errors))
    return canonical, max_error


def validate_vector_floorplan(
    plan: CanonicalFloorplan, taxonomy: FloorplanTaxonomy, *,
    tolerances: FEVPTolerances,
    quantizer: CoordinateQuantizer | None = None,
) -> list[str]:
    """Apply canonical checks plus explicit planar/CAD consistency rules."""

    errors = validate_floorplan(plan, taxonomy, tolerances=tolerances)
    footprint = Polygon(plan.footprint)
    footprint_valid = footprint.is_valid
    if not footprint_valid:
        errors.append("footprint has invalid ring topology")
    room_polygons = []
    for room in plan.rooms:
        polygon = Polygon(room.polygon, room.holes)
        room_polygons.append((room.id, polygon))
        if not polygon.is_valid:
            errors.append(f"room {room.id} has invalid hole topology")
            continue
        if footprint_valid:
            outside = polygon.difference(footprint).area
            if outside > tolerances.room_overlap_area_px2:
                errors.append(f"room {room.id} lies outside the footprint")
    for index, (left_id, left) in enumerate(room_polygons):
        for right_id, right in room_polygons[index + 1 :]:
            if not left.is_valid or not right.is_valid:
                continue
            if left.intersection(right).area > tolerances.room_overlap_area_px2:
                errors.append(f"rooms {left_id} and {right_id} overlap")
    walls = {wall.id: LineString([wall.start, wall.end]) for wall in plan.walls}
    for opening in plan.openings:
        host = walls.get(opening.wall_id)
        if host is not None and host.distance(Point(opening.center)) > tolerances.edge_snap_px:
            errors.append(f"opening {opening.id} is detached from its host wall")
    derived_junctions = derive_junctions(
        plan.walls,
        quantizer or CoordinateQuantizer(),
        tolerances.wall_junction_cluster_px,
    )
    for junction in plan.junctions:
        candidates = [
            item for item in derived_junctions
            if Point(junction.point).distance(Point(item.point)) <= tolerances.wall_junction_cluster_px
        ]
        if not candidates:
            errors.append(f"junction {junction.id} has no matching wall cluster")
        elif junction.degree != min(candidates, key=lambda item: Point(junction.point).distance(Point(item.point))).degree:
            errors.append(f"junction {junction.id} degree does not match wall incidence")
    return errors


def _ring_tokens(
    kind: str,
    ring: tuple[tuple[float, float], ...],
    quantizer: CoordinateQuantizer,
    *,
    delta_coordinates: bool,
) -> list[FloorplanToken]:
    quantized = [
        (quantizer.quantize(x), quantizer.quantize(y))
        for x, y in ring
    ]
    tokens = [FloorplanToken(f"{kind}_BEGIN")]
    if quantized:
        tokens.append(
            FloorplanToken("VERTEX", tuple(map(str, quantized[0])))
        )
    for previous, current in pairwise(quantized):
        if delta_coordinates:
            tokens.append(
                FloorplanToken(
                    "VD",
                    (
                        str(current[0] - previous[0]),
                        str(current[1] - previous[1]),
                    ),
                )
            )
        else:
            tokens.append(
                FloorplanToken("VERTEX", tuple(map(str, current)))
            )
    tokens.append(FloorplanToken("RING_END"))
    return tokens


def tokenize_floorplan(
    plan: CanonicalFloorplan, taxonomy: FloorplanTaxonomy, *,
    quantizer: CoordinateQuantizer | None = None,
    tolerances: FEVPTolerances | None = None,
    delta_coordinates: bool = True,
) -> TokenizedFloorplan:
    """Encode one canonical object into an ordered autoregressive grammar."""
    quantizer = quantizer or CoordinateQuantizer()
    canonical, max_error = canonicalize_floorplan(
        plan,
        taxonomy,
        quantizer=quantizer,
        tolerances=tolerances,
    )
    tokens = [
        FloorplanToken("SCHEMA", (TOKEN_SCHEMA_VERSION,)),
        FloorplanToken("CASE", (_atom(canonical.case_id),)),
        FloorplanToken("TRACK", (canonical.track,)),
        FloorplanToken("SIZE", (str(canonical.width), str(canonical.height))),
        FloorplanToken("SOURCE_HASH", (_atom(canonical.source_hash),)),
        FloorplanToken("QUANTIZER", (str(quantizer.scale),)),
        FloorplanToken(
            "COORDINATES",
            ("DELTA" if delta_coordinates else "ABSOLUTE",),
        ),
    ]
    tokens.extend(
        _ring_tokens(
            "FOOTPRINT",
            canonical.footprint,
            quantizer,
            delta_coordinates=delta_coordinates,
        )
    )
    for operation in canonical.operations:
        tokens.append(
            FloorplanToken(
                "OPERATION",
                (_atom(operation.kind), _atom(operation.target_id), _atom(operation.detail)),
            )
        )
    for room in canonical.rooms:
        tokens.append(
            FloorplanToken(
                "ROOM_BEGIN",
                (_atom(room.id), _atom(room.label), str(quantizer.quantize(room.confidence))),
            )
        )
        tokens.extend(
            _ring_tokens(
                "EXTERIOR",
                room.polygon,
                quantizer,
                delta_coordinates=delta_coordinates,
            )
        )
        for hole in room.holes:
            tokens.extend(
                _ring_tokens(
                    "HOLE",
                    hole,
                    quantizer,
                    delta_coordinates=delta_coordinates,
                )
            )
        if room.support:
            tokens.extend(
                _ring_tokens(
                    "SUPPORT",
                    room.support,
                    quantizer,
                    delta_coordinates=delta_coordinates,
                )
            )
        tokens.append(FloorplanToken("ROOM_END"))
    tokens.extend(
        FloorplanToken("ADJACENCY", (_atom(left), _atom(right)))
        for left, right in canonical.adjacency
    )
    tokens.extend(
        FloorplanToken(
            "WALL",
            (
                _atom(wall.id),
                str(quantizer.quantize(wall.start[0])),
                str(quantizer.quantize(wall.start[1])),
                str(quantizer.quantize(wall.end[0])),
                str(quantizer.quantize(wall.end[1])),
                str(quantizer.quantize(wall.thickness)),
                _atom(wall.wall_type),
            ),
        )
        for wall in canonical.walls
    )
    tokens.extend(
        FloorplanToken(
            "JUNCTION",
            (
                _atom(junction.id),
                str(quantizer.quantize(junction.point[0])),
                str(quantizer.quantize(junction.point[1])),
                str(junction.degree),
            ),
        )
        for junction in canonical.junctions
    )
    tokens.extend(
        FloorplanToken(
            "OPENING",
            (
                _atom(opening.id),
                _atom(opening.label),
                _atom(opening.wall_id),
                str(quantizer.quantize(opening.center[0])),
                str(quantizer.quantize(opening.center[1])),
                str(quantizer.quantize(opening.width)),
                str(quantizer.quantize(opening.confidence)),
            ),
        )
        for opening in canonical.openings
    )
    for item in canonical.furniture:
        tokens.append(
            FloorplanToken(
                "FURNITURE_BEGIN",
                (
                    _atom(item.id),
                    _atom(item.label),
                    _atom(item.room_id),
                    str(quantizer.quantize(item.confidence)),
                ),
            )
        )
        tokens.extend(
            _ring_tokens(
                "POLYGON",
                item.polygon,
                quantizer,
                delta_coordinates=delta_coordinates,
            )
        )
        tokens.append(FloorplanToken("FURNITURE_END"))
    tokens.append(FloorplanToken("END"))
    return TokenizedFloorplan(
        plan=canonical,
        tokens=tuple(tokens),
        exact_quantization=max_error <= 1.0e-12,
        max_quantization_error=max_error,
    )


def decode_floorplan(*args, **kwargs) -> CanonicalFloorplan:
    """Decode through the strict codec while preserving the public import path."""

    from floorflow.representation.unified_floorplan_token_codec import (
        decode_floorplan as decode,
    )

    return decode(*args, **kwargs)


def render_floorplan_rasters(*args, **kwargs) -> dict[str, object]:
    """Render through the derived-raster module without a circular import."""

    from floorflow.representation.vector_floorplan_raster import (
        render_floorplan_rasters as render,
    )

    return render(*args, **kwargs)


__all__ = [
    "TOKEN_SCHEMA_VERSION",
    "CoordinateQuantizer",
    "FloorplanToken",
    "TokenizedFloorplan",
    "canonicalize_floorplan",
    "decode_floorplan",
    "parse_tokens",
    "render_floorplan_rasters",
    "serialize_tokens",
    "tokenize_floorplan",
    "validate_vector_floorplan",
]
