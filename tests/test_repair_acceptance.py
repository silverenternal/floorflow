from floorflow.evaluation.fpeval_bridge import to_fpeval_plan
from floorflow.inference.repair_acceptance import (
    guarded_repair_decision,
    guarded_summary_decision,
)
from floorflow.geometry.wall_graph_readout import RasterWallPlan, ReadoutWall
from fpeval import evaluate


def _plan(with_duplicate: bool) -> RasterWallPlan:
    walls = [
        ReadoutWall((2.0, 2.0), (30.0, 2.0), 2.0),
        ReadoutWall((30.0, 2.0), (30.0, 30.0), 2.0),
        ReadoutWall((30.0, 30.0), (2.0, 30.0), 2.0),
        ReadoutWall((2.0, 30.0), (2.0, 2.0), 2.0),
    ]
    if with_duplicate:
        walls.append(ReadoutWall((30.0, 2.0), (30.0, 30.0), 2.0))
    rooms = [
        {
            "label": "room:Bedroom",
            "walls": [f"wall_{index + 1}" for index in range(4)],
            "polygon": [[2, 2], [30, 2], [30, 30], [2, 30]],
        }
    ]
    return RasterWallPlan(32.0, walls, rooms, [])


def test_repair_acceptance_accepts_lower_edit_cost_and_rejects_regression() -> None:
    target = to_fpeval_plan(_plan(False))
    raw = evaluate(to_fpeval_plan(_plan(True)), target)
    candidate = evaluate(to_fpeval_plan(_plan(False)), target)
    accepted = guarded_repair_decision(raw, candidate)
    assert accepted.accepted
    assert accepted.deltas["edit_cost"] < 0
    reverse = guarded_repair_decision(candidate, raw)
    assert not reverse.accepted


def test_summary_acceptance_rejects_metric_tradeoff() -> None:
    raw = {
        "room_f1": 0.5,
        "opening_f1": 0.5,
        "wall_f1": 0.5,
        "validity": 1.0,
        "edit_cost": 10.0,
    }
    candidate = {
        **raw,
        "room_f1": 0.49,
        "edit_cost": 5.0,
    }
    decision = guarded_summary_decision(raw, candidate)
    assert not decision.accepted
    assert "room_f1" in decision.reason
