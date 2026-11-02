from __future__ import annotations

import torch

from floorflow.inference.typed_junction_stage import typed_junction_rows
from floorflow.inference.verified_repair_loop import run_verified_repair_loop


def test_typed_junctions_cluster_endpoints_and_incidence() -> None:
    endpoints = torch.tensor(
        [
            [[0.1, 0.1], [0.8, 0.1]],
            [[0.8, 0.1], [0.8, 0.8]],
            [[0.1, 0.1], [0.1, 0.8]],
        ]
    )
    rows = typed_junction_rows(
        endpoints,
        torch.ones(3, dtype=torch.bool),
        frame=100.0,
        tolerance=0.02,
    )
    assert len(rows) == 4
    assert max(row["degree"] for row in rows) == 2
    assert all("incident_walls" in row for row in rows)


def test_verified_repair_loop_rejects_regression_and_freezes_best() -> None:
    def propose(state: dict[str, float], step: int):
        return {"value": state["value"] + (1.0 if step == 0 else -2.0)}

    def verify(state: dict[str, float]) -> float:
        return state["value"]

    result = run_verified_repair_loop(
        {"value": 0.0},
        propose,
        verify,
        max_steps=2,
    )
    assert result.state["value"] == 1.0
    assert result.score == 1.0
    assert [step.accepted for step in result.trace] == [True, False]
