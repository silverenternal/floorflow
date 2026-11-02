"""Strict decoder for the unified floorplan token grammar."""

from __future__ import annotations

from collections.abc import Iterable
from urllib.parse import unquote

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.canonical_floorplan import (
    CanonicalFloorplan,
    CanonicalFurniture,
    CanonicalJunction,
    CanonicalOpening,
    CanonicalOperation,
    CanonicalRoom,
    CanonicalWall,
)
from floorflow.evaluation.fevp_metric_contracts import FEVPTolerances
from floorflow.representation.unified_floorplan_tokens import (
    TOKEN_SCHEMA_VERSION,
    CoordinateQuantizer,
    FloorplanToken,
    canonicalize_floorplan,
    parse_tokens,
)


def _unatom(value: str) -> str:
    return unquote(value)


def _expect(
    token: FloorplanToken,
    kind: str,
    arity: int | None = None,
) -> tuple[str, ...]:
    if token.kind != kind:
        raise ValueError(f"expected {kind}, found {token.kind}")
    if arity is not None and len(token.args) != arity:
        raise ValueError(f"{kind} expects {arity} fields, found {len(token.args)}")
    return token.args


def _parse_ring(
    tokens: tuple[FloorplanToken, ...],
    index: int,
    prefix: str,
    quantizer: CoordinateQuantizer,
    coordinate_mode: str | None,
) -> tuple[tuple[tuple[float, float], ...], int]:
    _expect(tokens[index], f"{prefix}_BEGIN", 0)
    index += 1
    points = []
    current = None
    while index < len(tokens) and tokens[index].kind in {
        "VERTEX",
        "VD",
    }:
        if tokens[index].kind == "VERTEX":
            if coordinate_mode == "DELTA" and points:
                raise ValueError(
                    f"{prefix} uses VERTEX after the DELTA origin"
                )
            x, y = _expect(tokens[index], "VERTEX", 2)
            current = (int(x), int(y))
        else:
            if coordinate_mode == "ABSOLUTE":
                raise ValueError(f"{prefix} uses VD in ABSOLUTE mode")
            dx, dy = _expect(tokens[index], "VD", 2)
            if current is None:
                raise ValueError("VD requires a preceding VERTEX")
            current = (current[0] + int(dx), current[1] + int(dy))
        points.append(
            (
                quantizer.dequantize(current[0]),
                quantizer.dequantize(current[1]),
            )
        )
        index += 1
    if index >= len(tokens):
        raise ValueError(f"{prefix} is missing RING_END")
    _expect(tokens[index], "RING_END", 0)
    if len(points) < 3:
        raise ValueError(f"{prefix} needs at least three vertices")
    return tuple(points), index + 1


def decode_floorplan(
    tokens: str | Iterable[FloorplanToken],
    taxonomy: FloorplanTaxonomy,
    *,
    quantizer: CoordinateQuantizer | None = None,
    tolerances: FEVPTolerances | None = None,
) -> CanonicalFloorplan:
    """Decode the grammar; malformed or partial plans fail closed."""

    quantizer = quantizer or CoordinateQuantizer()
    tolerances = tolerances or FEVPTolerances()
    sequence = parse_tokens(tokens) if isinstance(tokens, str) else tuple(tokens)
    if not sequence or sequence[-1].kind != "END":
        raise ValueError("token sequence must terminate with END")
    index = 0
    schema = _expect(sequence[index], "SCHEMA", 1)[0]
    if schema != TOKEN_SCHEMA_VERSION:
        raise ValueError(f"unsupported token schema: {schema}")
    index += 1
    case_id = _unatom(_expect(sequence[index], "CASE", 1)[0])
    index += 1
    track = _expect(sequence[index], "TRACK", 1)[0]
    index += 1
    width, height = map(int, _expect(sequence[index], "SIZE", 2))
    index += 1
    source_hash = _unatom(_expect(sequence[index], "SOURCE_HASH", 1)[0])
    index += 1
    coordinate_mode = None
    if sequence[index].kind == "QUANTIZER":
        encoded_scale = int(_expect(sequence[index], "QUANTIZER", 1)[0])
        if encoded_scale <= 0:
            raise ValueError("quantizer scale must be positive")
        quantizer = CoordinateQuantizer(scale=encoded_scale)
        index += 1
    if sequence[index].kind == "COORDINATES":
        coordinate_mode = _expect(sequence[index], "COORDINATES", 1)[0]
        if coordinate_mode not in {"DELTA", "ABSOLUTE"}:
            raise ValueError(f"unsupported coordinate mode: {coordinate_mode}")
        index += 1
    footprint, index = _parse_ring(
        sequence,
        index,
        "FOOTPRINT",
        quantizer,
        coordinate_mode,
    )
    operations = []
    rooms = []
    adjacency = []
    walls = []
    junctions = []
    openings = []
    furniture = []
    while index < len(sequence) and sequence[index].kind != "END":
        token = sequence[index]
        if token.kind == "OPERATION":
            kind, target, detail = _expect(token, "OPERATION", 3)
            operations.append(
                CanonicalOperation(_unatom(kind), _unatom(target), _unatom(detail))
            )
            index += 1
        elif token.kind == "ROOM_BEGIN":
            room_id, label, confidence = _expect(token, "ROOM_BEGIN", 3)
            index += 1
            polygon, index = _parse_ring(
                sequence,
                index,
                "EXTERIOR",
                quantizer,
                coordinate_mode,
            )
            holes = []
            support = None
            while sequence[index].kind in {"HOLE_BEGIN", "SUPPORT_BEGIN"}:
                if sequence[index].kind == "HOLE_BEGIN":
                    hole, index = _parse_ring(
                        sequence,
                        index,
                        "HOLE",
                        quantizer,
                        coordinate_mode,
                    )
                    holes.append(hole)
                else:
                    support, index = _parse_ring(
                        sequence,
                        index,
                        "SUPPORT",
                        quantizer,
                        coordinate_mode,
                    )
            _expect(sequence[index], "ROOM_END", 0)
            index += 1
            rooms.append(
                CanonicalRoom(
                    id=_unatom(room_id),
                    label=_unatom(label),
                    polygon=polygon,
                    support=support,
                    holes=tuple(holes),
                    confidence=quantizer.dequantize(int(confidence)),
                )
            )
        elif token.kind == "ADJACENCY":
            adjacency.append(
                tuple(map(_unatom, _expect(token, "ADJACENCY", 2)))
            )
            index += 1
        elif token.kind == "WALL":
            wall_id, sx, sy, ex, ey, thickness, wall_type = _expect(
                token, "WALL", 7
            )
            walls.append(
                CanonicalWall(
                    id=_unatom(wall_id),
                    start=(
                        quantizer.dequantize(int(sx)),
                        quantizer.dequantize(int(sy)),
                    ),
                    end=(
                        quantizer.dequantize(int(ex)),
                        quantizer.dequantize(int(ey)),
                    ),
                    thickness=quantizer.dequantize(int(thickness)),
                    wall_type=_unatom(wall_type),
                )
            )
            index += 1
        elif token.kind == "JUNCTION":
            junction_id, x, y, degree = _expect(token, "JUNCTION", 4)
            junctions.append(
                CanonicalJunction(
                    id=_unatom(junction_id),
                    point=(
                        quantizer.dequantize(int(x)),
                        quantizer.dequantize(int(y)),
                    ),
                    degree=int(degree),
                )
            )
            index += 1
        elif token.kind == "OPENING":
            (
                opening_id,
                label,
                wall_id,
                x,
                y,
                width_value,
                confidence,
            ) = _expect(token, "OPENING", 7)
            openings.append(
                CanonicalOpening(
                    id=_unatom(opening_id),
                    label=_unatom(label),
                    center=(
                        quantizer.dequantize(int(x)),
                        quantizer.dequantize(int(y)),
                    ),
                    width=quantizer.dequantize(int(width_value)),
                    wall_id=_unatom(wall_id),
                    confidence=quantizer.dequantize(int(confidence)),
                )
            )
            index += 1
        elif token.kind == "FURNITURE_BEGIN":
            item_id, label, room_id, confidence = _expect(
                token, "FURNITURE_BEGIN", 4
            )
            index += 1
            polygon, index = _parse_ring(
                sequence,
                index,
                "POLYGON",
                quantizer,
                coordinate_mode,
            )
            _expect(sequence[index], "FURNITURE_END", 0)
            index += 1
            furniture.append(
                CanonicalFurniture(
                    id=_unatom(item_id),
                    label=_unatom(label),
                    polygon=polygon,
                    room_id=_unatom(room_id),
                    confidence=quantizer.dequantize(int(confidence)),
                )
            )
        else:
            raise ValueError(f"unexpected token in floorplan body: {token.kind}")
    if index != len(sequence) - 1:
        raise ValueError("tokens appear after END")
    plan = CanonicalFloorplan(
        case_id=case_id,
        track=track,
        width=width,
        height=height,
        footprint=footprint,
        rooms=tuple(rooms),
        walls=tuple(walls),
        openings=tuple(openings),
        furniture=tuple(furniture),
        adjacency=tuple(adjacency),
        source_hash=source_hash,
        junctions=tuple(junctions),
        operations=tuple(operations),
    )
    canonical, _ = canonicalize_floorplan(
        plan,
        taxonomy,
        quantizer=quantizer,
        tolerances=tolerances,
    )
    return canonical


__all__ = ["decode_floorplan"]
