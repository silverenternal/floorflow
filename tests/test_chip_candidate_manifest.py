from __future__ import annotations

import torch

from floorflow.chip_flow.candidate_consensus import CandidateConsensusFuser
from floorflow.chip_flow.candidate_manifest import candidate_manifest_from_sources


def _row(name: str, shift: float, cost: float) -> dict:
    boxes = [
        [0.0 + shift, 0.0, 1.0, 1.0],
        [1.0 + shift, 0.0, 1.0, 1.0],
    ]
    return {
        "index": 3,
        "raw_xywh": boxes,
        "legalized_xywh": boxes,
        "legalized_feasible": True,
        "legalized_hard_cost": cost,
        "raw_hard_cost": cost + 0.1,
        "gt_boxes": [[0.0, 0.0, 1.0, 1.0], [1.0, 0.0, 1.0, 1.0]],
        "areas": [1.0, 1.0],
        "constraints": [[0, 0, 0, 1, 0], [0, 0, 0, 1, 0]],
        "b2b": [[0.0, 1.0, 1.0]],
        "p2b": [[0.0, 0.0, 1.0]],
        "pins": [[0.2, 0.5]],
        "n_b2b": 1,
        "n_p2b": 1,
        "n_pins": 1,
        "source": name,
    }


def test_candidate_manifest_merges_sources_and_builds_target_free_features() -> None:
    cases = candidate_manifest_from_sources(
        {
            "checkpoint-b": [_row("b", 0.2, 1.0)],
            "checkpoint-a": [_row("a", 0.0, 1.1)],
        },
        reference_source="checkpoint-a",
    )
    assert len(cases) == 1
    case = cases[0]
    assert case["index"] == 3
    assert len(case["candidates"]) == 2
    assert case["candidates"][1]["quality"] == 1.0
    assert case["candidates"][0]["source"] == "checkpoint-a"
    assert len(case["candidates"][0]["features"]) == 14
    assert len(case["target_state"]) == 2
    assert case["source_geometry"]["areas"] == [1.0, 1.0]
    assert case["source_geometry"]["constraints"][0] == [0.0, 0.0, 0.0, 1.0, 0.0]


def test_candidate_manifest_rejects_source_input_mismatch() -> None:
    left = _row("a", 0.0, 1.1)
    right = _row("b", 0.2, 1.0)
    right["areas"] = [1.0, 2.0]
    try:
        candidate_manifest_from_sources(
            {"a": [left], "b": [right]},
        )
    except ValueError as error:
        assert "mismatches areas" in str(error)
    else:
        raise AssertionError("mismatched source inputs must fail closed")


def test_candidate_manifest_accepts_evaluator_rows_without_count_fields() -> None:
    row = _row("a", 0.0, 1.1)
    for key in ("n_b2b", "n_p2b", "n_pins"):
        del row[key]
    cases = candidate_manifest_from_sources({"a": [row]}, reference_source="a")
    assert len(cases[0]["candidates"][0]["features"]) == 14


def test_candidate_consensus_top_k_masks_non_selected_candidates() -> None:
    model = CandidateConsensusFuser(8, candidate_feature_dim=3, top_k=1)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    states = torch.tensor(
        [[[[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]]]]
    )
    features = torch.zeros(1, 3, 3)
    fused, weights = model(states, features)
    assert torch.equal(weights.sum(dim=-1), torch.ones(1))
    assert torch.isfinite(fused).all()
