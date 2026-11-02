"""Deterministic stage-exit audit for a canonical CAD floorplan."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from floorflow.evaluation.canonical_floorplan import (
    CanonicalFloorplan,
    CanonicalOperation,
)

CAD_STAGE_ORDER = (
    "envelope",
    "program",
    "rooms",
    "shared_boundaries",
    "walls",
    "junctions",
    "openings",
    "furniture",
    "validation",
)

STAGE_CONSUMERS = {
    "envelope": "footprint and boundary conditioning",
    "program": "room set and class identities",
    "rooms": "owner partition and canonical room polygons",
    "shared_boundaries": "room adjacency and owner-network wall candidates",
    "walls": "canonical wall network and opening hosts",
    "junctions": "wall incidence and junction validation",
    "openings": "hosted canonical openings",
    "furniture": "optional room-assigned furniture",
    "validation": "canonical token round trip and plan legality",
}


@dataclass(frozen=True)
class StageExit:
    name: str
    status: str
    gate: float | None
    consumer: str

    @property
    def completed(self) -> bool:
        return self.status in {"complete", "not_applicable"}


def audit_cad_stages(
    plan: CanonicalFloorplan,
    *,
    gates: Mapping[str, float] | None = None,
) -> tuple[StageExit, ...]:
    """Return one deterministic completion state for every CAD stage."""

    gates = gates or {}
    required = {
        "envelope": (
            len(plan.footprint) >= 3 and len(plan.rooms) > 0
        ),
        "program": len(plan.rooms) > 0,
        "rooms": len(plan.rooms) > 0,
        "shared_boundaries": len(plan.walls) > 0,
        "walls": len(plan.walls) > 0,
        "junctions": len(plan.junctions) > 0,
        "openings": len(plan.openings) > 0,
        "furniture": None,
        "validation": True,
    }
    states = []
    for name in CAD_STAGE_ORDER:
        completed = required[name]
        status = (
            "not_applicable"
            if completed is None
            else "complete"
            if completed
            else "incomplete"
        )
        states.append(
            StageExit(
                name=name,
                status=status,
                gate=(
                    float(gates[name])
                    if name in gates and gates[name] is not None
                    else None
                ),
                consumer=STAGE_CONSUMERS[name],
            )
        )
    return tuple(states)


def stage_exit_operations(
    states: tuple[StageExit, ...],
) -> tuple[CanonicalOperation, ...]:
    """Serialize stage states into the plan operation log."""

    return tuple(
        CanonicalOperation(
            "cad_stage_exit",
            state.name,
            (
                f"status={state.status};consumer={state.consumer};"
                f"gate={state.gate:.4f}"
                if state.gate is not None
                else f"status={state.status};consumer={state.consumer}"
            ),
        )
        for state in states
    )


__all__ = [
    "CAD_STAGE_ORDER",
    "STAGE_CONSUMERS",
    "StageExit",
    "audit_cad_stages",
    "stage_exit_operations",
]
