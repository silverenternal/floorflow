"""Convert the existing vector readout object into the unified CAD schema."""

from __future__ import annotations

from collections.abc import Sequence

from shapely.geometry import Polygon

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.canonical_floorplan import (
    TRACKS,
    CanonicalFloorplan,
    CanonicalFurniture,
    CanonicalOpening,
    CanonicalOperation,
    CanonicalRoom,
    CanonicalWall,
)
from floorflow.evaluation.fevp_metric_contracts import FEVPTolerances
from floorflow.geometry.wall_graph_readout import RasterWallPlan


def canonical_plan_from_readout(
    plan: RasterWallPlan,
    taxonomy: FloorplanTaxonomy,
    *,
    case_id: str,
    track: str,
    width: int,
    height: int,
    footprint: Sequence[Sequence[float]],
    source_hash: str,
    operations: tuple[CanonicalOperation, ...] = (),
    tolerances: FEVPTolerances | None = None,
) -> CanonicalFloorplan:
    """Preserve readout polygons and cross-references without rerasterizing."""

    if track not in TRACKS:
        raise ValueError(f"unknown floorplan track: {track}")
    tolerances = tolerances or FEVPTolerances()
    if not plan.rooms:
        raise ValueError("readout plan has no room polygons")
    room_ids = [f"room_{index + 1}" for index in range(len(plan.rooms))]
    rooms = []
    polygons = []
    for index, (room_id, room) in enumerate(zip(room_ids, plan.rooms, strict=True)):
        label = str(room["label"])
        if not taxonomy.is_room(label):
            raise ValueError(f"room {index} has illegal label {label}")
        polygon = tuple(tuple(map(float, point)) for point in room["polygon"])
        holes = tuple(
            tuple(tuple(map(float, point)) for point in hole)
            for hole in room.get("holes", ())
        )
        shape = Polygon(polygon, holes)
        if not shape.is_valid:
            raise ValueError(f"room {room_id} has invalid polygon geometry")
        rooms.append(CanonicalRoom(room_id, label, polygon, holes=holes))
        polygons.append(shape)
    walls = tuple(
        CanonicalWall(
            id=f"wall_{index + 1}",
            start=(float(wall.start[0]), float(wall.start[1])),
            end=(float(wall.end[0]), float(wall.end[1])),
            thickness=float(wall.thickness),
            wall_type=str(wall.wall_type),
        )
        for index, wall in enumerate(plan.walls)
    )
    wall_ids = [wall.id for wall in walls]
    openings = tuple(
        CanonicalOpening(
            id=f"opening_{index + 1}",
            label=str(opening.kind),
            center=(float(opening.center[0]), float(opening.center[1])),
            width=float(opening.width),
            wall_id=wall_ids[opening.wall_index],
        )
        for index, opening in enumerate(plan.openings)
        if 0 <= opening.wall_index < len(walls)
    )
    if len(openings) != len(plan.openings):
        raise ValueError("readout opening references an invalid wall index")
    furniture = tuple(
        CanonicalFurniture(
            id=f"furniture_{index + 1}",
            label=str(item.label),
            polygon=tuple(tuple(map(float, point)) for point in item.polygon),
            room_id=room_ids[item.room_index],
            confidence=float(item.confidence),
        )
        for index, item in enumerate(plan.furniture)
        if 0 <= item.room_index < len(rooms)
    )
    if len(furniture) != len(plan.furniture):
        raise ValueError("readout furniture references an invalid room index")
    adjacency = []
    for left in range(len(polygons)):
        for right in range(left + 1, len(polygons)):
            contact = polygons[left].intersection(polygons[right])
            if (
                polygons[left].distance(polygons[right])
                <= tolerances.edge_contact_px
                and contact.length > 1.0e-6
            ):
                adjacency.append((room_ids[left], room_ids[right]))
    return CanonicalFloorplan(
        case_id=case_id,
        track=track,
        width=int(width),
        height=int(height),
        footprint=tuple(tuple(map(float, point)) for point in footprint),
        rooms=tuple(rooms),
        walls=walls,
        openings=openings,
        furniture=furniture,
        adjacency=tuple(adjacency),
        source_hash=source_hash,
        operations=operations,
    )


__all__ = ["canonical_plan_from_readout"]
