from __future__ import annotations

import torch

from floorflow.chip_flow.candidate_legalizability import (
    CandidateLegalizabilityReranker,
    listwise_ranking_loss,
    select_candidate,
    target_free_candidate_features,
)


def _entry(
    *,
    v_rel: float,
    cost: float,
    b2b: float,
    compactness: float,
) -> dict:
    return {
        "stages": {
            "legalized": {
                "v_rel": v_rel,
                "cost": cost,
                "boundary": v_rel,
                "grouping": v_rel,
            }
        },
        "displacement": {"center_l2": b2b},
        "target_free_geometry": {
            "b2b_hpwl": b2b,
            "p2b_hpwl": b2b,
            "outline_compactness": compactness,
            "outline_aspect": -0.1,
        },
        "legalizer": {"rung": 2, "solves": 4},
    }


def test_candidate_features_ignore_official_cost() -> None:
    left = _entry(v_rel=0.1, cost=1.0, b2b=2.0, compactness=0.8)
    changed = {**left, "stages": {"legalized": {**left["stages"]["legalized"], "cost": 9.0}}}
    assert torch.equal(
        target_free_candidate_features([left]),
        target_free_candidate_features([changed]),
    )


def test_candidate_features_follow_candidate_permutation() -> None:
    entries = [
        _entry(v_rel=0.1, cost=1.0, b2b=2.0, compactness=0.8),
        _entry(v_rel=0.2, cost=1.2, b2b=1.0, compactness=0.9),
    ]
    features = target_free_candidate_features(entries)
    permuted = target_free_candidate_features([entries[1], entries[0]])
    assert torch.equal(permuted, features.flip(0))


def test_listwise_loss_prefers_lower_scores_for_lower_costs() -> None:
    costs = torch.tensor([1.0, 1.5])
    good = listwise_ranking_loss(torch.tensor([1.0, 0.0]), costs)
    bad = listwise_ranking_loss(torch.tensor([0.0, 1.0]), costs)
    assert good < bad


def test_reranker_selects_minimum_score() -> None:
    model = CandidateLegalizabilityReranker(hidden=4)
    features = torch.rand(3, model.feature_dim)
    with torch.no_grad():
        scores = model(features)
    assert select_candidate(scores) == int(torch.argmin(scores).item())
