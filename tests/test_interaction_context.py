"""Tests for observable multi-step interaction context."""

from __future__ import annotations

from floorflow.data.interaction_context import enrich_sequence_context


def _candidate(candidate_id: str) -> dict:
    return {
        "candidate_id": candidate_id,
        "role": "positive",
    }


def test_sequence_context_encodes_remaining_typed_actions() -> None:
    groups = [
        {
            "sequence_id": "val:case",
            "step": 0,
            "operation": "room_add",
            "parent_candidate_id": None,
            "candidates": [_candidate("root")],
        },
        {
            "sequence_id": "val:case",
            "step": 1,
            "operation": "room_remove",
            "parent_candidate_id": "root",
            "candidates": [_candidate("child")],
        },
        {
            "sequence_id": "val:case",
            "step": 2,
            "operation": "opening_delete",
            "parent_candidate_id": "child",
            "candidates": [_candidate("leaf")],
        },
    ]
    summary = enrich_sequence_context(groups)
    assert summary["candidates_augmented"] == 3
    root = groups[0]["candidates"][0]["sequence_features"]
    assert root[0] == 0.0
    assert root[1] == 1.0
    assert root[2:7] == [0.0, 1.0, 0.0, 0.0, 0.0]
    assert root[7:12] == [0.0, 0.0, 0.0, 1.0, 0.0]
    assert groups[2]["candidates"][0]["sequence_features"][1] == 0.0
