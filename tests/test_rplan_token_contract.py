from __future__ import annotations

from floorflow.data.floorplan_token_targets import (
    build_floorplan_token_vocabulary,
    training_token_sequence,
)
from floorflow.data.rplan_boundary_adapter import (
    RPlanBoundaryDataset,
    rplan_taxonomy,
)
from floorflow.evaluation.rplan_token_contract import (
    audit_rplan_token_contract,
)


def _row() -> dict:
    return {
        "name": "token-contract",
        "boundary": [[0, 0], [0, 100], [100, 100], [100, 0]],
        "rooms": [
            {
                "id": 0,
                "category": 0,
                "location": [25, 50],
                "r_boundary": [[0, 0], [50, 0], [50, 100], [0, 100]],
            }
        ],
        "adjacencies": [],
    }


def test_rplan_token_contract_matches_plan_entities() -> None:
    taxonomy = rplan_taxonomy()
    dataset = RPlanBoundaryDataset(
        "unused.json",
        taxonomy,
        rows=[_row()],
    )
    plan = dataset._plan(0)
    vocabulary, _statistics = build_floorplan_token_vocabulary(
        [training_token_sequence(plan, taxonomy)]
    )
    result = audit_rplan_token_contract(
        plan,
        taxonomy,
        vocabulary,
        max_length=512,
    )
    assert result["passed"]
    assert result["entity_count_match"]
    assert result["oov_count"] == 0


def test_rplan_token_contract_detects_length_overflow() -> None:
    taxonomy = rplan_taxonomy()
    dataset = RPlanBoundaryDataset(
        "unused.json",
        taxonomy,
        rows=[_row()],
    )
    plan = dataset._plan(0)
    vocabulary, _statistics = build_floorplan_token_vocabulary(
        [training_token_sequence(plan, taxonomy)]
    )
    result = audit_rplan_token_contract(
        plan,
        taxonomy,
        vocabulary,
        max_length=8,
    )
    assert not result["passed"]
    assert result["length_overflow"]
