"""Canonical vector floorplan schema shared by evaluation and tokenization."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from shapely.geometry import Polygon

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.fevp_metric_contracts import FEVPTolerances

SCHEMA_VERSION = "sota-floorplan-output-v2"
TRACKS = (
    "rplan_generation",
    "cubicasa_reconstruction",
    "typed_editing",
    "free_text_editing",
)


@dataclass(frozen=True)
class CanonicalRoom:
    id: str
    label: str
    polygon: tuple[tuple[float, float], ...]
    support: tuple[tuple[float, float], ...] | None = None
    holes: tuple[tuple[tuple[float, float], ...], ...] = ()
    confidence: float = 1.0


@dataclass(frozen=True)
class CanonicalWall:
    id: str
    start: tuple[float, float]
    end: tuple[float, float]
    thickness: float
    wall_type: str = "interior"


@dataclass(frozen=True)
class CanonicalOpening:
    id: str
    label: str
    center: tuple[float, float]
    width: float
    wall_id: str
    confidence: float = 1.0

    @property
    def family(self) -> str:
        return "window" if self.label.startswith("window:") else "door"


@dataclass(frozen=True)
class CanonicalFurniture:
    id: str
    label: str
    polygon: tuple[tuple[float, float], ...]
    room_id: str
    confidence: float = 1.0


@dataclass(frozen=True)
class CanonicalJunction:
    id: str
    point: tuple[float, float]
    degree: int


@dataclass(frozen=True)
class CanonicalOperation:
    """Explicit representation-changing operation applied before decoding."""

    kind: str
    target_id: str
    detail: str = ""


@dataclass(frozen=True)
class CanonicalFloorplan:
    case_id: str
    track: str
    width: int
    height: int
    footprint: tuple[tuple[float, float], ...]
    rooms: tuple[CanonicalRoom, ...]
    walls: tuple[CanonicalWall, ...]
    openings: tuple[CanonicalOpening, ...]
    furniture: tuple[CanonicalFurniture, ...]
    adjacency: tuple[tuple[str, str], ...]
    source_hash: str
    schema_version: str = SCHEMA_VERSION
    junctions: tuple[CanonicalJunction, ...] = ()
    operations: tuple[CanonicalOperation, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["adjacency"] = [list(edge) for edge in sorted(self.adjacency)]
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CanonicalFloorplan:
        return cls(
            case_id=str(payload["case_id"]),
            track=str(payload["track"]),
            width=int(payload["width"]),
            height=int(payload["height"]),
            footprint=tuple(tuple(map(float, point)) for point in payload["footprint"]),
            rooms=tuple(
                CanonicalRoom(
                    id=str(room["id"]),
                    label=str(room["label"]),
                    polygon=tuple(
                        tuple(map(float, point)) for point in room["polygon"]
                    ),
                    support=(
                        tuple(
                            tuple(map(float, point))
                            for point in room["support"]
                        )
                        if room.get("support")
                        else None
                    ),
                    holes=tuple(
                        tuple(tuple(map(float, point)) for point in ring)
                        for ring in room.get("holes", ())
                    ),
                    confidence=float(room.get("confidence", 1.0)),
                )
                for room in payload["rooms"]
            ),
            walls=tuple(
                CanonicalWall(
                    id=str(wall["id"]),
                    start=tuple(map(float, wall["start"])),
                    end=tuple(map(float, wall["end"])),
                    thickness=float(wall["thickness"]),
                    wall_type=str(wall.get("wall_type", "interior")),
                )
                for wall in payload["walls"]
            ),
            openings=tuple(
                CanonicalOpening(
                    id=str(opening["id"]),
                    label=str(opening["label"]),
                    center=tuple(map(float, opening["center"])),
                    width=float(opening["width"]),
                    wall_id=str(opening["wall_id"]),
                    confidence=float(opening.get("confidence", 1.0)),
                )
                for opening in payload["openings"]
            ),
            furniture=tuple(
                CanonicalFurniture(
                    id=str(item["id"]),
                    label=str(item["label"]),
                    polygon=tuple(
                        tuple(map(float, point)) for point in item["polygon"]
                    ),
                    room_id=str(item["room_id"]),
                    confidence=float(item.get("confidence", 1.0)),
                )
                for item in payload["furniture"]
            ),
            adjacency=tuple(
                tuple(sorted((str(edge[0]), str(edge[1]))))
                for edge in payload["adjacency"]
            ),
            source_hash=str(payload["source_hash"]),
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
            junctions=tuple(
                CanonicalJunction(
                    id=str(junction["id"]),
                    point=tuple(map(float, junction["point"])),
                    degree=int(junction["degree"]),
                )
                for junction in payload.get("junctions", ())
            ),
            operations=tuple(
                CanonicalOperation(
                    kind=str(operation["kind"]),
                    target_id=str(operation["target_id"]),
                    detail=str(operation.get("detail", "")),
                )
                for operation in payload.get("operations", ())
            ),
        )


def _finite_points(
    points: Sequence[Sequence[float]],
    name: str,
    errors: list[str],
) -> tuple[tuple[float, float], ...]:
    parsed = tuple(tuple(map(float, point)) for point in points)
    if len(parsed) >= 2 and parsed[0] == parsed[-1]:
        parsed = parsed[:-1]
    if len(parsed) < 3:
        errors.append(f"{name} needs at least three distinct vertices")
    if any(
        len(point) != 2 or not all(math.isfinite(value) for value in point)
        for point in parsed
    ):
        errors.append(f"{name} contains a non-finite or non-2D point")
    return parsed


def validate_floorplan(
    plan: CanonicalFloorplan,
    taxonomy: FloorplanTaxonomy,
    *,
    tolerances: FEVPTolerances,
    require_complete_cad: bool = False,
) -> list[str]:
    """Validate the single output contract shared by every SOTA track."""

    errors: list[str] = []
    if plan.schema_version != SCHEMA_VERSION:
        errors.append(f"schema_version must equal {SCHEMA_VERSION}")
    if plan.track not in TRACKS:
        errors.append(f"unknown track: {plan.track}")
    if not plan.case_id or not plan.source_hash:
        errors.append("case_id and source_hash must be non-empty")
    if plan.width <= 0 or plan.height <= 0:
        errors.append("width and height must be positive")
    footprint = _finite_points(plan.footprint, "footprint", errors)
    if footprint and not Polygon(footprint).is_valid:
        errors.append("footprint is self-intersecting")
    room_ids = [room.id for room in plan.rooms]
    wall_ids = [wall.id for wall in plan.walls]
    furniture_ids = [item.id for item in plan.furniture]
    opening_ids = [opening.id for opening in plan.openings]
    junction_ids = [junction.id for junction in plan.junctions]
    if len(set(room_ids)) != len(room_ids):
        errors.append("room ids must be unique")
    if len(set(wall_ids)) != len(wall_ids):
        errors.append("wall ids must be unique")
    if len(set(opening_ids)) != len(opening_ids):
        errors.append("opening ids must be unique")
    if len(set(furniture_ids)) != len(furniture_ids):
        errors.append("furniture ids must be unique")
    if len(set(junction_ids)) != len(junction_ids):
        errors.append("junction ids must be unique")
    for room in plan.rooms:
        points = _finite_points(room.polygon, f"room {room.id}", errors)
        if not taxonomy.is_room(room.label):
            errors.append(f"room {room.id} has illegal label {room.label}")
        if points and not Polygon(points).is_valid:
            errors.append(f"room {room.id} polygon is self-intersecting")
        for hole_index, hole in enumerate(room.holes):
            hole_points = _finite_points(
                hole,
                f"room {room.id} hole {hole_index}",
                errors,
            )
            if hole_points and not Polygon(hole_points).is_valid:
                errors.append(f"room {room.id} hole {hole_index} is self-intersecting")
        if room.support:
            support = _finite_points(
                room.support,
                f"room {room.id} support",
                errors,
            )
            if support and not Polygon(support).is_valid:
                errors.append(f"room {room.id} support is self-intersecting")
        if require_complete_cad and room.support is None:
            errors.append(f"room {room.id} is missing support geometry")
    for wall in plan.walls:
        if not all(math.isfinite(value) for value in (*wall.start, *wall.end)):
            errors.append(f"wall {wall.id} has a non-finite endpoint")
        if math.dist(wall.start, wall.end) <= tolerances.wall_junction_cluster_px:
            errors.append(f"wall {wall.id} is too short")
        if not math.isfinite(wall.thickness) or wall.thickness < 0.05:
            errors.append(f"wall {wall.id} has non-positive thickness")
    for opening in plan.openings:
        if opening.wall_id not in set(wall_ids):
            errors.append(f"opening {opening.id} references an unknown wall")
        if not taxonomy.is_opening(opening.label):
            errors.append(f"opening {opening.id} has illegal label")
        if not all(math.isfinite(value) for value in opening.center):
            errors.append(f"opening {opening.id} has a non-finite center")
        if not math.isfinite(opening.width) or opening.width <= 0.0:
            errors.append(f"opening {opening.id} needs positive width")
    for junction in plan.junctions:
        if junction.degree <= 0:
            errors.append(f"junction {junction.id} needs positive degree")
        if not all(math.isfinite(value) for value in junction.point):
            errors.append(f"junction {junction.id} has a non-finite point")
    for item in plan.furniture:
        if not taxonomy.is_furniture(item.label):
            errors.append(f"furniture {item.id} has illegal label")
        if item.room_id not in room_ids:
            errors.append(f"furniture {item.id} references an unknown room")
        points = _finite_points(
            item.polygon,
            f"furniture {item.id}",
            errors,
        )
        if points and not Polygon(points).is_valid:
            errors.append(f"furniture {item.id} polygon is self-intersecting")
    seen_edges = set()
    for left, right in plan.adjacency:
        if left == right:
            errors.append("room adjacency cannot contain a self-loop")
        if left not in room_ids or right not in room_ids:
            errors.append("adjacency references an unknown room")
        edge = tuple(sorted((left, right)))
        if edge in seen_edges:
            errors.append(f"duplicate adjacency edge: {edge}")
        seen_edges.add(edge)
    return errors


def canonical_fingerprint(plan: CanonicalFloorplan) -> str:
    payload = json.dumps(
        plan.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "SCHEMA_VERSION",
    "TRACKS",
    "CanonicalFloorplan",
    "CanonicalFurniture",
    "CanonicalJunction",
    "CanonicalOpening",
    "CanonicalOperation",
    "CanonicalRoom",
    "CanonicalWall",
    "canonical_fingerprint",
    "validate_floorplan",
]
