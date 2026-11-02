"""Tests for serialized finite CAD-stage completion states."""

from dataclasses import replace

from floorflow.evaluation.cad_stage_audit import (
    CAD_STAGE_ORDER,
    audit_cad_stages,
    stage_exit_operations,
)
from floorflow.evaluation.canonical_floorplan import (
    CanonicalFloorplan,
    CanonicalJunction,
    CanonicalOpening,
    CanonicalRoom,
    CanonicalWall,
)


def _plan() -> CanonicalFloorplan:
    return CanonicalFloorplan(
        case_id="stage-audit",
        track="rplan_generation",
        width=8,
        height=8,
        footprint=((0, 0), (8, 0), (8, 8), (0, 8)),
        rooms=(
            CanonicalRoom(
                "room_01",
                "room:LivingRoom",
                ((0, 0), (4, 0), (4, 8), (0, 8)),
            ),
            CanonicalRoom(
                "room_02",
                "room:Kitchen",
                ((4, 0), (8, 0), (8, 8), (4, 8)),
            ),
        ),
        walls=(
            CanonicalWall("wall_01", (4, 0), (4, 8), 3.0),
            CanonicalWall("wall_02", (0, 0), (8, 0), 3.0, "exterior"),
        ),
        openings=(
            CanonicalOpening(
                "opening_01",
                "door:Door",
                (4, 2),
                2,
                "wall_01",
            ),
        ),
        furniture=(),
        adjacency=(("room_01", "room_02"),),
        source_hash="stage-audit",
        junctions=(
            CanonicalJunction("junction_01", (4, 0), 3),
        ),
    )


def test_every_stage_has_exit_state_consumer_and_gate() -> None:
    states = audit_cad_stages(
        _plan(),
        gates={"walls": 0.75, "openings": 0.5},
    )

    assert tuple(state.name for state in states) == CAD_STAGE_ORDER
    assert all(state.consumer for state in states)
    assert all(state.completed for state in states)
    assert next(state for state in states if state.name == "furniture").status == (
        "not_applicable"
    )
    assert next(state for state in states if state.name == "walls").gate == 0.75


def test_missing_stage_output_fails_closed_in_audit() -> None:
    plan = replace(_plan(), walls=(), openings=())

    states = audit_cad_stages(plan)

    assert any(not state.completed for state in states)
    operations = stage_exit_operations(states)
    assert len(operations) == len(CAD_STAGE_ORDER)
    assert any(
        operation.target_id == "walls"
        and operation.detail.startswith("status=incomplete")
        for operation in operations
    )
