from __future__ import annotations

import pytest
import torch

from floorflow.chip_flow.hierarchical_advance import (
    advance_hierarchy_state,
    coarse_commit_threshold,
    coarse_stability_masks,
)
from floorflow.chip_flow.hierarchical_flow import (
    HierarchicalTokenFlowModel,
    build_hierarchy_targets,
)
from floorflow.chip_flow.hierarchy_state import (
    HierarchyPrediction,
    HierarchyState,
)
from floorflow.chip_flow.legalizer_portfolio import (
    LegalizerCandidate,
    select_legalizer_candidate,
)
from floorflow.chip_flow.portfolio_router import (
    PortfolioRouter,
    router_features,
    validate_router_accuracy,
)
from floorflow.chip_flow.relation_gate_balance import relation_gate_regularization
from floorflow.chip_flow.source_mixing import source_shelf_probability_at_step
from floorflow.chip_flow.sparse_layers import (
    StateEdgeRelay,
    prepare_state_edge_cache,
)
from floorflow.chip_flow.spatial_occupancy_coupler import SpatialOccupancyCoupler
from floorflow.chip_flow.structural_relations import (
    StructuralRelationHead,
    structural_relation_features,
)
from floorflow.chip_flow.target_free_geometry import target_free_geometry_metrics
from floorflow.chip_flow.training_runtime import (
    make_optimizer,
    unfreeze_backbone_at_step,
)
from floorflow.chip_flow.typed_graph_encoder import (
    TypedCrossAttention,
    TypedGraphEncoder,
)


def _constraints(batch: int = 1, count: int = 6) -> torch.Tensor:
    constraints = torch.zeros(batch, count, 5)
    constraints[..., 3] = torch.tensor([1, 1, 2, 2, 3, 3])
    constraints[:, :2, 4] = 1
    return constraints


def _batch() -> dict:
    count = 6
    boxes = torch.zeros(2, count, 4)
    boxes[..., 2:] = 1.0
    boxes[..., 0] = torch.arange(count) * 1.2
    return {
        "areas": boxes[..., 2] * boxes[..., 3],
        "boxes": boxes,
        "constraints": _constraints(batch=2),
        "features": torch.zeros(2, count, 19),
        "target_latent": torch.randn(2, count, 3),
        "b2b": torch.tensor([[[0.0, 1.0, 1.0], [1.0, 2.0, 0.5]]] * 2),
        "p2b": torch.tensor([[[0.0, 0.0, 1.0], [1.0, 1.0, 0.5]]] * 2),
        "pins": torch.rand(2, 2, 2),
        "n_b2b": torch.tensor([2, 2]),
        "n_p2b": torch.tensor([2, 2]),
        "n_pins": torch.tensor([2, 2]),
    }


def test_relation_attention_masks_cross_membership_leakage() -> None:
    torch.manual_seed(3)
    module = TypedCrossAttention(16, 4, relation_specific=True)
    module.gate.data.fill_(1.0)
    module.relation_gate.data.copy_(torch.tensor([0.0, 1.0, 0.0, 0.0]))
    query = torch.randn(1, 2, 16)
    memory = torch.randn(1, 2, 16)
    masks = (
        torch.ones(1, 2, 1, dtype=torch.bool),
        torch.tensor([[[True, False], [False, True]]]),
        torch.zeros(1, 2, 1, dtype=torch.bool),
        torch.zeros(1, 2, 1, dtype=torch.bool),
    )
    relation_memory = torch.randn(1, 2, 16)
    original = module.forward_relations(
        query,
        memory,
        torch.ones(1, 2, 2, dtype=torch.bool),
        (relation_memory, relation_memory, relation_memory, relation_memory),
        masks,
    )
    changed_memory = memory.clone()
    changed_memory[:, 0] += 5.0
    changed_relation_memory = relation_memory.clone()
    changed_relation_memory[:, 0] += 5.0
    changed = module.forward_relations(
        query,
        changed_memory,
        torch.ones(1, 2, 2, dtype=torch.bool),
        (
            changed_relation_memory,
            changed_relation_memory,
            changed_relation_memory,
            changed_relation_memory,
        ),
        masks,
    )
    torch.testing.assert_close(original[:, 1], changed[:, 1])
    assert not torch.equal(original[:, 0], changed[:, 0])
    assert torch.isfinite(original).all()


def test_incidence_normalized_relation_scales_active_edges() -> None:
    module = TypedCrossAttention(
        16,
        4,
        relation_specific=True,
        incidence_normalized_relations=True,
    )
    mixed = torch.ones(1, 2, 16)
    allowed = torch.ones(1, 2, 1, 4, dtype=torch.bool)
    normalized = module._normalize_relation(mixed, allowed)
    torch.testing.assert_close(normalized, torch.full_like(mixed, 0.5))


def test_structural_relation_targets_capture_group_connectivity() -> None:
    boxes = torch.tensor(
        [[[0.0, 0.0, 1.0, 1.0], [1.0, 0.0, 1.0, 1.0],
          [3.0, 0.0, 1.0, 1.0], [5.0, 0.0, 1.0, 1.0],
          [0.0, 2.0, 2.0, 1.0], [2.0, 2.0, 1.0, 1.0]]]
    )
    areas = boxes[..., 2] * boxes[..., 3]
    features, active = structural_relation_features(
        boxes,
        areas,
        _constraints(),
    )
    assert active[0, 0] and active[0, 1]
    assert features[0, 0, 6] == 0.0
    assert features[0, 1, 6] > 0.0
    assert features[0, 1, 5] > features[0, 0, 5]


def test_structural_relation_features_are_scale_equivariant() -> None:
    boxes = torch.tensor(
        [[[0.0, 0.0, 1.0, 1.0], [1.0, 0.0, 1.0, 1.0],
          [3.0, 0.0, 1.0, 1.0], [5.0, 0.0, 1.0, 1.0],
          [0.0, 2.0, 2.0, 1.0], [2.0, 2.0, 1.0, 1.0]]]
    )
    areas = boxes[..., 2] * boxes[..., 3]
    original, original_active = structural_relation_features(
        boxes,
        areas,
        _constraints(),
    )
    scaled, scaled_active = structural_relation_features(
        boxes * 10.0,
        areas * 100.0,
        _constraints(),
    )

    torch.testing.assert_close(original, scaled, atol=1.0e-5, rtol=1.0e-5)
    torch.testing.assert_close(original_active, scaled_active)


def test_structural_relation_head_and_loss_are_differentiable() -> None:
    torch.manual_seed(5)
    head = StructuralRelationHead(16, bottleneck=8)
    head.gate.data.fill_(1.0)
    torch.nn.init.normal_(head.feature_head[-1].weight, std=0.02)
    torch.nn.init.normal_(head.token_head[-1].weight, std=0.02)
    tokens = torch.randn(1, 6, 16, requires_grad=True)
    state = torch.randn(1, 6, 3, requires_grad=True)
    membership = torch.zeros(1, 6, 9, dtype=torch.bool)
    membership[0, :2, 0] = True
    membership[0, 2:4, 1] = True
    structural_mask = membership.any(dim=1)
    structural_state = torch.randn(1, 9, 3)
    features, tokens_residual = head(
        tokens,
        state,
        torch.ones(1, 6),
        membership,
        structural_mask,
        structural_state,
    )
    assert features.shape == (1, 9, 7)
    assert tokens_residual.shape == (1, 9, 16)
    loss = features.square().mean() + tokens_residual.square().mean()
    loss.backward()
    assert tokens.grad is not None and torch.isfinite(tokens.grad).all()
    assert state.grad is not None and torch.isfinite(state.grad).all()


def test_coarse_commit_freezes_l0_l1_but_not_l2() -> None:
    state = HierarchyState(
        global_state=torch.zeros(2, 4),
        structural_state=torch.zeros(2, 3, 3),
        block_state=torch.zeros(2, 5, 3),
        port_state=torch.zeros(2, 2, 2),
    )
    prediction = HierarchyPrediction(
        global_state=torch.ones(2, 4),
        structural_state=torch.ones(2, 3, 3),
        block_state=torch.ones(2, 5, 3),
        port_state=torch.ones(2, 2, 2),
    )
    advanced = advance_hierarchy_state(
        state,
        prediction,
        torch.full((2,), 0.5),
        step_size=0.25,
        port_mask=torch.ones(2, 2, dtype=torch.bool),
        coarse_frozen=True,
    )
    torch.testing.assert_close(advanced.global_state, state.global_state)
    torch.testing.assert_close(advanced.structural_state, state.structural_state)
    assert not torch.equal(advanced.block_state, state.block_state)
    assert coarse_commit_threshold(0.3, 10) == 3
    assert coarse_commit_threshold(0.3, 1) == 1


def test_confidence_gated_commit_uses_a_bounded_anchor() -> None:
    state = HierarchyState(
        global_state=torch.zeros(1, 4),
        structural_state=torch.zeros(1, 2, 3),
        block_state=torch.zeros(1, 3, 3),
        port_state=torch.zeros(1, 1, 2),
    )
    prediction = HierarchyPrediction(
        global_state=torch.full((1, 4), 0.02),
        structural_state=torch.stack(
            [
                torch.full((1, 3), 0.02),
                torch.ones(1, 3),
            ],
            dim=1,
        ),
        block_state=torch.ones(1, 3, 3),
        port_state=torch.ones(1, 1, 2),
    )
    masks = coarse_stability_masks(
        state,
        prediction,
        tolerance=0.05,
        structural_mask=torch.tensor([[True, True]]),
    )
    advanced = advance_hierarchy_state(
        state,
        prediction,
        torch.ones(1),
        step_size=1.0,
        port_mask=torch.ones(1, 1, dtype=torch.bool),
        coarse_anchor_masks=masks,
        coarse_anchor_strength=0.25,
        coarse_anchor_max_step=0.1,
    )
    torch.testing.assert_close(
        advanced.global_state,
        torch.full((1, 4), 0.005),
    )
    torch.testing.assert_close(
        advanced.structural_state[:, 0],
        torch.full((1, 3), 0.005),
    )
    torch.testing.assert_close(
        advanced.structural_state[:, 1],
        torch.ones(1, 3),
    )


def test_route_aware_spatial_field_is_sensitive_to_pin_demand() -> None:
    torch.manual_seed(7)
    batch, count, hidden = 1, 4, 16
    areas = torch.ones(batch, count)
    state = torch.randn(batch, count, 3)
    tokens = torch.randn(batch, count, hidden)
    b2b = torch.tensor([[[0.0, 1.0, 1.0]]])
    p2b = torch.tensor([[[0.0, 0.0, 1.0]]])
    pins = torch.tensor([[[0.1, 0.2]]])
    cache = prepare_state_edge_cache(
        areas=areas,
        b2b=b2b,
        p2b=p2b,
        pins=pins,
        n_b2b=torch.tensor([1]),
        n_p2b=torch.tensor([1]),
        n_pins=torch.tensor([1]),
    )
    module = SpatialOccupancyCoupler(hidden, route_aware=True)
    module.gate.data.fill_(1.0)
    first, _ = module(tokens, state, areas, cache=cache)
    changed_pins = torch.tensor([[[0.9, 0.8]]])
    changed_cache = {
        **cache,
        "pins": changed_pins.reshape(-1, 2),
    }
    second, _ = module(tokens, state, areas, cache=changed_cache)
    assert not torch.equal(first, second)


def test_state_edge_relay_is_equivariant_to_uniform_physical_scaling() -> None:
    torch.manual_seed(17)
    relay = StateEdgeRelay()
    relay.gate.data.fill_(1.0)
    state = torch.randn(1, 4, 3)
    areas = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    b2b = torch.tensor([[[0.0, 1.0, 1.0], [2.0, 3.0, 0.5]]])
    p2b = torch.tensor([[[0.0, 0.0, 1.0], [1.0, 2.0, 0.5]]])
    pins = torch.tensor([[[10.0, 20.0], [30.0, 40.0]]])
    first_cache = prepare_state_edge_cache(
        areas=areas,
        b2b=b2b,
        p2b=p2b,
        pins=pins,
        n_b2b=torch.tensor([2]),
        n_p2b=torch.tensor([2]),
        n_pins=torch.tensor([2]),
    )
    second_cache = prepare_state_edge_cache(
        areas=areas * 100.0,
        b2b=b2b,
        p2b=p2b,
        pins=pins * 10.0,
        n_b2b=torch.tensor([2]),
        n_p2b=torch.tensor([2]),
        n_pins=torch.tensor([2]),
    )

    first = relay(state, first_cache)
    second = relay(state, second_cache)

    torch.testing.assert_close(first, second, atol=1.0e-5, rtol=1.0e-5)


def test_target_free_geometry_and_balanced_selector_protect_b2b() -> None:
    boxes = torch.tensor(
        [[0.0, 0.0, 1.0, 1.0], [1.0, 0.0, 1.0, 1.0], [3.0, 0.0, 1.0, 1.0]]
    )
    metrics = target_free_geometry_metrics(
        boxes,
        torch.ones(3),
        torch.tensor([[0.0, 1.0, 1.0]]),
        torch.tensor([[0.0, 0.0, 1.0]]),
        torch.tensor([[0.0, 0.0]]),
        1,
        1,
    )
    assert metrics["outline_compactness"] > 0.0
    baseline = LegalizerCandidate("baseline", 0.10, 0.2, 1.0, 1.0, 1, 1, 1.0, 1.0, 0.5, 0.2)
    candidate = LegalizerCandidate("candidate", 0.105, 0.2, 1.0, 1.0, 1, 1, 0.5, 0.5, 0.8, 0.0)
    assert select_legalizer_candidate(
        baseline,
        (candidate,),
        policy="legality_first",
    ).name == "baseline"
    assert select_legalizer_candidate(
        baseline,
        (candidate,),
        policy="balanced",
    ).name == "candidate"


def test_portfolio_router_shapes_and_gradients() -> None:
    constraints = _constraints(batch=2)
    features = router_features(
        constraints,
        torch.zeros(2, 2, 3),
        torch.zeros(2, 1, 3),
        torch.tensor([2, 4]),
        torch.tensor([1, 3]),
        torch.ones(2, 6),
        torch.rand(2, 3, 2),
        torch.tensor([3, 3]),
    )
    router = PortfolioRouter(8)
    logits = router(features)
    assert logits.shape == (2, 3)
    logits.square().mean().backward()
    assert router.network[-1].weight.grad is not None
    with pytest.raises(ValueError, match="below"):
        validate_router_accuracy(0.48, 0.65)
    validate_router_accuracy(0.48, 0.65, allow_low_accuracy=True)


def test_complexity_conditioning_changes_global_token() -> None:
    torch.manual_seed(8)
    encoder = TypedGraphEncoder(
        feature_dim=19,
        hidden=16,
        heads=4,
        graph_layers=1,
        complexity_conditioning=True,
    )
    torch.nn.init.normal_(encoder.complexity.projection.weight, std=0.02)
    features = torch.randn(1, 6, 19)
    constraints = _constraints()
    b2b = torch.tensor([[[0.0, 1.0, 1.0], [1.0, 2.0, 0.5]]])
    p2b = torch.tensor([[[0.0, 0.0, 1.0]]])
    pins = torch.rand(1, 2, 2)
    first = encoder(
        features,
        constraints,
        areas=torch.ones(1, 6),
        b2b=b2b,
        p2b=p2b,
        pins=pins,
        n_b2b=torch.tensor([1]),
        n_p2b=torch.tensor([1]),
        n_pins=torch.tensor([2]),
    )
    second = encoder(
        features,
        constraints,
        areas=torch.ones(1, 6),
        b2b=b2b,
        p2b=p2b,
        pins=pins,
        n_b2b=torch.tensor([2]),
        n_p2b=torch.tensor([1]),
        n_pins=torch.tensor([2]),
    )
    assert not torch.equal(first.global_token, second.global_token)


def test_relation_gate_regularization_penalizes_collapse() -> None:
    module = TypedCrossAttention(16, 4, relation_specific=True)
    collapsed = relation_gate_regularization(module)
    module.relation_gate.data.fill_(0.5)
    balanced = relation_gate_regularization(module)
    assert collapsed > balanced


def test_integrated_v2_model_connects_all_cpu_paths() -> None:
    torch.manual_seed(11)
    count = 6
    boxes = torch.zeros(2, count, 4)
    boxes[..., 2:] = 1.0
    boxes[..., 0] = torch.arange(count) * 1.2
    batch = {
        "areas": boxes[..., 2] * boxes[..., 3],
        "boxes": boxes,
        "constraints": _constraints(batch=2),
        "features": torch.zeros(2, count, 19),
        "target_latent": torch.randn(2, count, 3),
        "b2b": torch.tensor(
            [[[0.0, 1.0, 1.0], [1.0, 2.0, 0.5]]] * 2
        ),
        "p2b": torch.tensor([[[0.0, 0.0, 1.0], [1.0, 1.0, 0.5]]] * 2),
        "pins": torch.rand(2, 2, 2),
        "n_b2b": torch.tensor([2, 2]),
        "n_p2b": torch.tensor([2, 2]),
        "n_pins": torch.tensor([2, 2]),
    }
    model = HierarchicalTokenFlowModel(
        hidden=32,
        heads=4,
        depth=1,
        head_dim=16,
        typed_layers=1,
        relation_specific=True,
        structural_relations=True,
        complexity_conditioning=True,
        coarse_commit_fraction=0.5,
        structured_heads=True,
        dynamic_graph_input=True,
        spatial_route_field=True,
        projector_grid_size=8,
    )
    context = model.prepare_context(batch)
    target = build_hierarchy_targets(batch, context)
    prediction = model(target, torch.full((2,), 0.5), context)
    assert prediction.structural_relations is not None
    assert prediction.legalizer_displacement is not None
    loss = prediction.block_state.square().mean()
    loss = loss + prediction.structural_relations.square().mean()
    loss.backward()
    assert model.relation_head is not None
    assert model.relation_head.feature_head[-1].weight.grad is not None


def test_staged_backbone_unfreeze_keeps_optimizer_membership() -> None:
    model = torch.nn.Module()
    model.backbone = torch.nn.Linear(3, 3)
    model.head = torch.nn.Linear(3, 1)
    model.backbone.weight.requires_grad_(False)
    model.backbone.bias.requires_grad_(False)
    optimizer = make_optimizer(
        model,
        lr=1.0e-3,
        weight_decay=0.0,
        fused=False,
        include_frozen=True,
    )
    optimizer_parameters = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert id(model.backbone.weight) in optimizer_parameters
    assert not unfreeze_backbone_at_step(model, 10, 9)
    assert unfreeze_backbone_at_step(model, 10, 10)
    assert model.backbone.weight.requires_grad


def test_source_shelf_probability_schedule_reaches_deployment_mix() -> None:
    args = type(
        "Args",
        (),
        {
            "hierarchy_source_shelf_probability": 0.25,
            "hierarchy_source_shelf_probability_final": 0.75,
            "hierarchy_source_shelf_probability_steps": 100,
            "weight_schedule": "linear",
        },
    )()
    assert source_shelf_probability_at_step(args, 0) == 0.25
    assert source_shelf_probability_at_step(args, 50) == 0.5
    assert source_shelf_probability_at_step(args, 100) == 0.75


