from __future__ import annotations

import torch

from floorflow.data.design_points import DesignPointSet, immutable_token_mask
from floorflow.evaluation.space_syntax import space_syntax_metrics
from floorflow.heads.codetree_anchor_decoder import CodeTreeAnchorDecoder
from floorflow.heads.floorplan_token_head import FloorplanTokenHead
from floorflow.heads.room_instance_token_head import RoomInstanceTokenHead
from floorflow.heads.token_graph_residual_head import TokenGraphResidualHead
from floorflow.inference.junction_face_readout import junction_face_candidate
from floorflow.inference.token_beam_search import (
    beam_search_floorplan_tokens,
    verifier_guided_beam_search,
)
from floorflow.objectives.token_grammar import FloorplanTokenVocabulary
from floorflow.objectives.token_grpo import group_relative_advantages, token_grpo_loss
from floorflow.objectives.token_reward import token_candidate_reward
from floorflow.objectives.typed_state_hierarchy_loss import (
    typed_state_hierarchy_loss,
)
from floorflow.objectives.verifier_policy import verifier_policy_loss
from floorflow.representation.typed_floorplan_state import TypedFloorplanState
from floorflow.representation.typed_stage_layers import stage_layers_dict
from floorflow.representation.unified_floorplan_tokens import FloorplanToken


def test_token_graph_residual_updates_discrete_graph() -> None:
    head = TokenGraphResidualHead(
        8,
        3,
        max_rooms=2,
        max_walls=4,
        max_openings=2,
        max_junctions=3,
    )
    predictions = {
        "room_exists": torch.zeros(1, 2),
        "room_class": torch.zeros(1, 2, 3),
        "wall_exists": torch.zeros(1, 4),
        "wall_exterior_logits": torch.zeros(1, 4),
        "opening_exists": torch.zeros(1, 2),
        "opening_host_logits": torch.zeros(1, 2, 4),
        "room_adjacency_logits": torch.zeros(1, 2, 2),
    }
    output = head(
        predictions,
        torch.randn(1, 6, 8),
        torch.tensor([[True, True, True, True, False, False]]),
        torch.tensor([[0, 2, 2, 3, 0, 0]]),
    )
    assert output["room_exists"].shape == (1, 2)
    assert output["token_junction_degree_logits"].shape == (1, 3, 9)
    assert 0.0 <= output["token_graph_gate"].item() <= 1.0
    assert output["token_junction_coordinates"].shape == (1, 3, 2)


def test_codetree_anchor_decoder_and_immutable_mask() -> None:
    decoder = CodeTreeAnchorDecoder(
        8,
        3,
        max_rooms=3,
        polygon_points=5,
        heads=4,
    )
    output = decoder(
        torch.randn(1, 8, 8),
        torch.ones(1, 8, dtype=torch.bool),
        torch.tensor([[0, 2, 2, 2, 3, 3, 5, 7]]),
        predictions={
            "room_class": torch.randn(1, 3, 4),
            "room_polygons": torch.rand(1, 3, 5, 2),
        },
    )
    assert output["codetree_polygons"].shape == (1, 3, 5, 2)
    assert output["codetree_room_codes"].shape == (1, 3)
    assert output["codetree_polygon_codes"].shape == (1, 3)
    assert torch.isfinite(output["codetree_vq_loss"])
    tokens = (
        FloorplanToken("ROOM_BEGIN", ("room_01",)),
        FloorplanToken("RING_END"),
        FloorplanToken("ROOM_END"),
        FloorplanToken("WALL", ("wall_01",)),
    )
    design = DesignPointSet(immutable_rooms=frozenset({"room_01"}))
    assert immutable_token_mask(tokens, design) == (True, True, True, False)


def test_room_instance_semantic_token_identity() -> None:
    head = RoomInstanceTokenHead(8, 3)
    features = torch.randn(1, 4, 8)
    logits = torch.randn(1, 4, 4)
    tokens = head(
        features,
        logits,
        torch.tensor([[True, True, False, False]]),
    )
    assert tokens.shape == (1, 4, 8)
    assert not tokens[0, 2:].any()


def test_typed_state_aligns_entities_with_hungarian_assignment() -> None:
    reference = TypedFloorplanState(
        room_exists=torch.ones(1, 2),
        room_polygons=torch.tensor(
            [
                [
                    [[0.0, 0.0], [0.1, 0.0], [0.1, 0.1]],
                    [[0.8, 0.8], [0.9, 0.8], [0.9, 0.9]],
                ]
            ]
        ),
    )
    candidate = TypedFloorplanState(
        room_exists=torch.ones(1, 2),
        room_polygons=reference.room_polygons.flip(1),
    )
    aligned = candidate.align_rooms_to(reference)
    assert torch.allclose(aligned.room_polygons, reference.room_polygons)
    layers = stage_layers_dict(aligned)
    assert set(layers) == {"room_instance", "topology", "geometry"}
    assert "polygons" not in layers["room_instance"]
    assert "room_polygons" in layers["geometry"]


def test_training_alignment_uses_soft_assignment_without_hungarian(
    monkeypatch,
) -> None:
    reference = TypedFloorplanState(
        room_exists=torch.ones(1, 2),
        room_polygons=torch.tensor(
            [
                [
                    [[0.0, 0.0], [0.1, 0.0], [0.1, 0.1]],
                    [[0.8, 0.8], [0.9, 0.8], [0.9, 0.9]],
                ]
            ]
        ),
    )
    candidate = TypedFloorplanState(
        room_exists=torch.ones(1, 2),
        room_polygons=reference.room_polygons.flip(1),
    )
    monkeypatch.setattr(
        "floorflow.representation.typed_floorplan_state.linear_sum_assignment",
        lambda _cost: (_ for _ in ()).throw(AssertionError("CPU assignment used")),
    )
    aligned = candidate.align_rooms_to(reference, differentiable=True)
    assert aligned.room_polygons is not None
    assert torch.isfinite(aligned.room_polygons).all()


def test_junction_faces_candidate_from_projected_walls() -> None:
    output = {
        "wall_endpoints": torch.tensor(
            [
                [
                    [[0.0, 0.0], [1.0, 0.0]],
                    [[1.0, 0.0], [1.0, 1.0]],
                    [[1.0, 1.0], [0.0, 1.0]],
                    [[0.0, 1.0], [0.0, 0.0]],
                ]
            ]
        ),
        "wall_exists": torch.full((1, 4), 10.0),
        "room_polygons": torch.tensor(
            [[[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]]]
        ),
        "token_junction_coordinates": torch.tensor(
            [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]]
        ),
        "token_junction_degree_logits": torch.nn.functional.one_hot(
            torch.tensor([[2, 2, 2, 2]]),
            num_classes=9,
        ).float(),
    }
    candidate = junction_face_candidate(output)
    assert candidate is not None
    assert candidate["room_polygons"].shape == output["room_polygons"].shape


def test_beam_search_and_space_syntax_reranking() -> None:
    vocabulary = FloorplanTokenVocabulary(("END", "FURNITURE_BEGIN"))
    head = FloorplanTokenHead(8, vocabulary.model_vocab_size, 8, hidden_dim=8)
    head.eval()
    context = torch.randn(1, 1, 8)
    prompt = (
        FloorplanToken("SCHEMA", ("v",)),
        FloorplanToken("CASE", ("<CASE>",)),
        FloorplanToken("TRACK", ("rplan_generation",)),
        FloorplanToken("SIZE", ("32", "32")),
        FloorplanToken("SOURCE_HASH", ("<SOURCE>",)),
        FloorplanToken("FURNITURE_END"),
    )
    candidates = beam_search_floorplan_tokens(
        head,
        context,
        torch.ones(1, 1, dtype=torch.bool),
        vocabulary,
        prompt,
        max_new_tokens=1,
        beam_width=2,
    )
    assert candidates
    best, _score = verifier_guided_beam_search(
        head,
        context,
        torch.ones(1, 1, dtype=torch.bool),
        vocabulary,
        prompt,
        lambda tokens: float(tokens[-1].kind == "END"),
        max_new_tokens=1,
        beam_width=2,
        immutable_reference=(*prompt, FloorplanToken("END")),
        immutable_mask=(False,) * len(prompt) + (True,),
    )
    assert best[-1].kind == "END"
    syntax = space_syntax_metrics(
        torch.tensor([[0, 1, 1], [1, 0, 0], [1, 0, 0]]).numpy(),
        ("room:LivingRoom", "room:Bedroom", "room:Bath"),
    )
    assert syntax["public_integration_advantage"] > 0.0


def test_grpo_group_advantages_and_loss() -> None:
    rewards = torch.tensor([0.0, 1.0, 0.5, 0.5])
    advantages = group_relative_advantages(rewards, 2)
    assert torch.allclose(advantages.reshape(2, 2).mean(dim=1), torch.zeros(2))
    logits = torch.randn(4, 3, 5, requires_grad=True)
    targets = torch.randint(0, 5, (4, 3))
    loss, components = token_grpo_loss(
        logits,
        targets,
        torch.ones(4, 3, dtype=torch.bool),
        rewards,
        group_size=2,
    )
    assert torch.isfinite(loss)
    assert "token_grpo_advantage_abs_mean" in components
    reward, reward_components = token_candidate_reward(
        {
            "contract_valid": True,
            "functional_watertight": True,
            "hole_count": 0,
            "room_count": 4,
        },
        {"public_integration_advantage": 0.2},
        expected_room_count=4,
    )
    assert reward > 0
    assert reward_components["reward_room_error"] == 0.0
    hierarchy = typed_state_hierarchy_loss(
        {
            "polygon": torch.tensor(1.0),
            "identity": torch.tensor(2.0),
            "wall": torch.tensor(3.0),
            "token": torch.tensor(4.0),
        },
        {"primary": 1.0, "topology": 0.5, "sequence": 0.2},
    )
    assert torch.isfinite(hierarchy.total)
    assert set(hierarchy.components) == {
        "hierarchy_primary_loss",
        "hierarchy_topology_loss",
        "hierarchy_sequence_loss",
    }
    policy_loss, policy_components = verifier_policy_loss(
        logits,
        targets,
        torch.ones(4, 3, dtype=torch.bool),
        torch.tensor([0.0, 1.0, 0.5, 0.2]),
    )
    assert torch.isfinite(policy_loss)
    assert "verifier_reward_mean" in policy_components


__all__ = []
