"""Tests for learned CAD wall-candidate scoring."""

import numpy as np
import torch

from floorflow.geometry.wall_graph_readout import ReadoutWall
from floorflow.heads.cad_wall_candidate_scorer import CadWallCandidateScorer
from floorflow.inference.cad_wall_candidate_runtime import candidate_walls


def test_candidate_scorer_shapes_and_gradients() -> None:
    scorer = CadWallCandidateScorer(16, hidden_dim=32, graph_dim=24)
    features = torch.randn(2, 16, 16, 16, requires_grad=True)
    endpoints = torch.rand(2, 5, 2, 2)
    valid = torch.tensor(
        [[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]],
        dtype=torch.bool,
    )
    evidence = torch.rand(2, 5)
    graph = torch.randn(2, 4, 24)

    output = scorer(features, endpoints, valid, evidence, graph)
    (output["score_logits"].mean() + output["endpoints"].mean()).backward()

    assert output["score_logits"].shape == (2, 5)
    assert torch.allclose(
        output["score_logits"],
        torch.logit(evidence.clamp(1.0e-4, 1.0 - 1.0e-4)),
        atol=1.0e-5,
    )
    assert output["exterior_logits"].shape == (2, 5)
    assert output["budget_count"].shape == (2,)
    assert output["endpoints"].shape == (2, 5, 2, 2)
    assert torch.all(output["score_residual_logits"].abs() <= 1.0)
    assert features.grad is not None


def test_encoded_score_consumes_local_candidate_features() -> None:
    scorer = CadWallCandidateScorer(
        8,
        hidden_dim=16,
        graph_dim=12,
        use_encoded_score=True,
    )
    assert scorer.score[0].in_features == 12 + 16
    with torch.no_grad():
        scorer.score[2].weight.fill_(0.1)
    features = torch.randn(1, 8, 8, 8, requires_grad=True)
    output = scorer(
        features,
        torch.rand(1, 3, 2, 2),
        torch.ones(1, 3, dtype=torch.bool),
        torch.full((1, 3), 0.5),
        torch.randn(1, 2, 12),
    )
    output["score_logits"].mean().backward()
    assert features.grad is not None
    assert bool(features.grad.abs().sum() > 0)


def test_wall_budget_heads_are_live_and_sample_dependent_on_step_one() -> None:
    scorer = CadWallCandidateScorer(8, hidden_dim=16, graph_dim=12)
    valid = torch.ones(2, 3, dtype=torch.bool)
    endpoints = torch.rand(2, 3, 2, 2)
    evidence = torch.full((2, 3), 0.5)
    graph = torch.randn(2, 2, 12)
    first = scorer(
        torch.randn(2, 8, 8, 8),
        endpoints,
        valid,
        evidence,
        graph,
    )
    second = scorer(
        torch.randn(2, 8, 8, 8),
        endpoints,
        valid,
        evidence,
        graph,
    )
    assert not torch.allclose(
        first["budget_fraction"],
        second["budget_fraction"],
    )
    loss = (
        first["budget_fraction"].sum()
        + first["exterior_budget_fraction"].sum()
        + first["interior_budget_fraction"].sum()
    )
    loss.backward()
    assert scorer.budget[0].weight.grad is not None
    assert scorer.budget[2].weight.grad is not None
    assert scorer.budget[0].weight.grad.abs().sum() > 0
    assert scorer.budget[2].weight.grad.abs().sum() > 0
    for head in scorer.phase_budget.values():
        assert head[0].weight.grad is not None
        assert head[2].weight.grad is not None
        assert head[0].weight.grad.abs().sum() > 0
        assert head[2].weight.grad.abs().sum() > 0


def test_candidate_walls_include_runtime_host_segments() -> None:
    extra = ReadoutWall(
        start=(0.0, 0.0),
        end=(7.0, 7.0),
        thickness=3.0,
        wall_type="interior",
    )
    walls, evidence = candidate_walls(
        [],
        torch.ones(8, 8).numpy(),
        torch.ones(8, 8).numpy(),
        extra_walls=[extra],
    )
    observed = {
        tuple(sorted((wall.start, wall.end)))
        for wall in walls
    }
    assert tuple(sorted((extra.start, extra.end))) in observed
    assert evidence.shape == (len(walls),)
    assert np.all(np.diff(evidence) <= 1.0e-6)


def test_candidate_source_embedding_is_trainable_and_sample_specific() -> None:
    scorer = CadWallCandidateScorer(
        8,
        hidden_dim=16,
        graph_dim=12,
        use_encoded_score=True,
        source_features=True,
    )
    with torch.no_grad():
        scorer.score[2].weight.fill_(0.1)
    features = torch.randn(1, 8, 8, 8)
    endpoints = torch.rand(1, 3, 2, 2)
    valid = torch.ones(1, 3, dtype=torch.bool)
    evidence = torch.full((1, 3), 0.5)
    graph = torch.randn(1, 2, 12)
    first = scorer(
        features,
        endpoints,
        valid,
        evidence,
        graph,
        proposal_source=torch.tensor([[0, 1, 2]]),
    )
    second = scorer(
        features,
        endpoints,
        valid,
        evidence,
        graph,
        proposal_source=torch.tensor([[3, 4, 5]]),
    )
    assert not torch.allclose(
        first["score_logits"],
        second["score_logits"],
    )
    first["score_logits"].mean().backward()
    assert scorer.source_embedding.weight.grad is not None
