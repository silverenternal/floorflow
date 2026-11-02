"""Design-point constraints for immutable CAD structure and editable tokens."""

from __future__ import annotations

from dataclasses import dataclass, field

from shapely.geometry import LineString, Point, Polygon

from floorflow.data.floorplan_token_targets import training_token_sequence
from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.canonical_floorplan import CanonicalFloorplan
from floorflow.representation.unified_floorplan_tokens import FloorplanToken


@dataclass(frozen=True)
class DesignPoint:
    kind: str
    target_id: str
    geometry: tuple[tuple[float, float], ...] = ()
    editable: bool = False


@dataclass(frozen=True)
class DesignPointSet:
    points: tuple[DesignPoint, ...] = ()
    immutable_rooms: frozenset[str] = field(default_factory=frozenset)
    immutable_walls: frozenset[str] = field(default_factory=frozenset)
    immutable_openings: frozenset[str] = field(default_factory=frozenset)

    def entity_editable(self, kind: str, target_id: str) -> bool:
        if kind == "room":
            return target_id not in self.immutable_rooms
        if kind == "wall":
            return target_id not in self.immutable_walls
        if kind == "opening":
            return target_id not in self.immutable_openings
        return True


def derive_design_points(
    plan: CanonicalFloorplan,
    edit_region: Polygon,
) -> DesignPointSet:
    """Preserve entities that do not intersect the declared editable region."""

    immutable_rooms = frozenset(
        room.id
        for room in plan.rooms
        if not Polygon(room.polygon, room.holes).intersects(edit_region)
    )
    immutable_walls = frozenset(
        wall.id
        for wall in plan.walls
        if edit_region.distance(LineString([wall.start, wall.end])) > 0.5
    )
    immutable_openings = frozenset(
        opening.id
        for opening in plan.openings
        if not edit_region.intersects(Point(opening.center).buffer(opening.width))
    )
    points = []
    for room in plan.rooms:
        points.append(
            DesignPoint(
                "room",
                room.id,
                tuple(room.polygon),
                room.id not in immutable_rooms,
            )
        )
    for wall in plan.walls:
        points.append(
            DesignPoint(
                "wall",
                wall.id,
                (wall.start, wall.end),
                wall.id not in immutable_walls,
            )
        )
    for opening in plan.openings:
        points.append(
            DesignPoint(
                "opening",
                opening.id,
                (opening.center,),
                opening.id not in immutable_openings,
            )
        )
    return DesignPointSet(
        tuple(points),
        immutable_rooms,
        immutable_walls,
        immutable_openings,
    )


def immutable_token_mask(
    tokens: tuple[FloorplanToken, ...],
    design: DesignPointSet,
) -> tuple[bool, ...]:
    """Mark tokens whose owning room/wall/opening must remain unchanged."""

    immutable = []
    current_entity: tuple[str, str] | None = None
    ring_depth = 0
    for token in tokens:
        if token.kind == "ROOM_BEGIN" and token.args:
            current_entity = ("room", token.args[0])
            immutable.append(not design.entity_editable(*current_entity))
            continue
        if token.kind == "WALL" and token.args:
            current_entity = ("wall", token.args[0])
            immutable.append(not design.entity_editable(*current_entity))
            current_entity = None
            continue
        if token.kind == "OPENING" and token.args:
            immutable.append(not design.entity_editable("opening", token.args[0]))
            continue
        if token.kind.endswith("_BEGIN"):
            ring_depth += 1
        if current_entity is not None:
            immutable.append(not design.entity_editable(*current_entity))
        else:
            immutable.append(False)
        if token.kind == "RING_END":
            ring_depth = max(0, ring_depth - 1)
        if token.kind == "ROOM_END":
            current_entity = None
    return tuple(immutable)


def build_edit_token_constraints(
    before_plan: CanonicalFloorplan,
    edit_region: Polygon,
    taxonomy: FloorplanTaxonomy,
) -> tuple[DesignPointSet, tuple[FloorplanToken, ...], tuple[bool, ...]]:
    design = derive_design_points(before_plan, edit_region)
    reference = training_token_sequence(before_plan, taxonomy)
    return design, reference, immutable_token_mask(reference, design)


__all__ = [
    "DesignPoint",
    "DesignPointSet",
    "build_edit_token_constraints",
    "derive_design_points",
    "immutable_token_mask",
]
