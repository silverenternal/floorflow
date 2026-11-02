from __future__ import annotations

from floorflow.geometry.wall_graph_readout import (
    RasterWallPlan,
    ReadoutOpening,
    ReadoutWall,
)
from floorflow.inference.cad_wall_candidate_runtime import (
    select_opening_safe_plan,
)


def _plan(center: tuple[float, float], wall_x: float) -> RasterWallPlan:
    return RasterWallPlan(
        frame=100.0,
        walls=[ReadoutWall((wall_x, 0.0), (wall_x, 100.0), 3.0)],
        rooms=[],
        openings=[
            ReadoutOpening(center, 10.0, "door", 0),
        ],
    )


def test_opening_safe_gate_accepts_unchanged_readout() -> None:
    control = _plan((20.0, 40.0), 20.0)
    candidate = _plan((20.0, 40.0), 25.0)
    assert select_opening_safe_plan(control, candidate) is candidate


def test_opening_safe_gate_falls_back_when_readout_changes() -> None:
    control = _plan((20.0, 40.0), 20.0)
    candidate = _plan((23.0, 40.0), 25.0)
    assert select_opening_safe_plan(control, candidate) is control
