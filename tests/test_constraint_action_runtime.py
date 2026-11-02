"""Tests for hard-masked single-trajectory action selection."""

from __future__ import annotations

from collections import Counter

import torch

from floorflow.data.taxonomy import build_taxonomy
from floorflow.inference.constraint_action_runtime import (
    ActionCandidate,
    ConstraintActionRuntime,
    hard_rule_selection,
)
from floorflow.model.constraint_aware_gate import ConstraintAwareGate


def _taxonomy():
    return build_taxonomy(
        {
            "room": Counter({"Bedroom": 3, "Kitchen": 3}),
            "furniture": Counter({"Sink": 3}),
            "door": Counter({"Swing Beside": 3}),
            "window": Counter({"Regular": 3}),
        },
        room_min_count=1,
        furniture_min_count=1,
    )


def _candidate(
    candidate_id: str,
    hard: bool,
    *,
    declared_legal: bool | None = None,
) -> ActionCandidate:
    nodes = len(_taxonomy().room_ids)
    declared_legal = hard if declared_legal is None else declared_legal
    return ActionCandidate(
        candidate_id=candidate_id,
        role="positive" if hard else "near_violation",
        node_features=torch.randn(nodes, 8),
        adjacency=torch.eye(nodes),
        global_features=torch.randn(8),
        constraint_diagnostics=torch.rand(7),
        action_features=torch.randn(15),
        hard_mask=torch.full((8,), hard, dtype=torch.bool),
        declared_legal=declared_legal,
        operation_id=0,
        utility=1.0 if hard else 0.0,
    )


def test_runtime_excludes_invalid_action_mass() -> None:
    taxonomy = _taxonomy()
    model = ConstraintAwareGate(hidden_dim=32, embedding_dim=16)
    runtime = ConstraintActionRuntime(model)
    candidates = [
        _candidate("invalid", False),
        _candidate("valid", True),
    ]
    decision = runtime.select(candidates, taxonomy)
    assert decision.candidate_id == "valid"
    assert decision.hard_feasible
    assert decision.invalid_probability_mass == 0.0


def test_runtime_zeroes_probability_for_raw_hard_declared_illegal() -> None:
    taxonomy = _taxonomy()
    model = ConstraintAwareGate(hidden_dim=32, embedding_dim=16)
    runtime = ConstraintActionRuntime(model)
    candidates = [
        _candidate(
            "typed_violation",
            True,
            declared_legal=False,
        ),
        _candidate("valid", True),
    ]
    decision = runtime.select(candidates, taxonomy)
    assert decision.candidate_id == "valid"
    assert decision.hard_feasible
    assert decision.invalid_probability_mass == 0.0


def test_hard_rule_selection_uses_only_feasible_candidates() -> None:
    candidates = [
        _candidate("invalid", False),
        _candidate("valid", True),
    ]
    decision = hard_rule_selection(candidates)
    assert decision.candidate_id == "valid"
    assert decision.selection_source == "hard_rule_hash"


def test_state_blind_runtime_still_obeys_exact_legal_mask() -> None:
    taxonomy = _taxonomy()
    model = ConstraintAwareGate(hidden_dim=32, embedding_dim=16)
    runtime = ConstraintActionRuntime(model, state_blind=True)
    candidates = [
        _candidate("typed_violation", True, declared_legal=False),
        _candidate("valid", True),
    ]
    decision = runtime.select(candidates, taxonomy)
    assert decision.candidate_id == "valid"
    assert decision.hard_feasible
    assert decision.invalid_probability_mass == 0.0
