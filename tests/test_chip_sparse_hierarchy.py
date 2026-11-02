from __future__ import annotations

import torch

from floorflow.chip_flow.flow_adapter import (
    count_parameters,
    parameter_matched_global_slot_config,
)
from floorflow.chip_flow.model import SparseAttention
from floorflow.chip_flow.sparse_flow_model import ChipSparseHierarchyFlowModel
from floorflow.chip_flow.sparse_hierarchy import SparseHierarchyEncoder


def _inputs() -> tuple[torch.Tensor, ...]:
    features = torch.randn(2, 7, 8)
    features[0, :, :5] = 0.0
    features[1, :, :5] = 1.0
    b2b = torch.tensor(
        [
            [
                [0.0, 1.0, 1.0],
                [1.0, 2.0, 0.5],
                [2.0, 3.0, 0.5],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            [
                [0.0, 1.0, 1.0],
                [1.0, 2.0, 1.0],
                [2.0, 3.0, 1.0],
                [3.0, 4.0, 1.0],
                [4.0, 5.0, 1.0],
                [5.0, 6.0, 1.0],
            ],
        ]
    )
    p2b = torch.tensor(
        [
            [[0.0, 0.0, 1.0]],
            [[0.0, 6.0, 1.0]],
        ]
    )
    pins = torch.tensor(
        [
            [[0.2, 0.4], [0.0, 0.0]],
            [[0.7, 0.8], [0.0, 0.0]],
        ]
    )
    n_b2b = torch.tensor([2, 6])
    n_p2b = torch.tensor([1, 1])
    n_pins = torch.tensor([1, 1])
    return features, b2b, p2b, pins, n_b2b, n_p2b, n_pins


def _model() -> SparseHierarchyEncoder:
    return SparseHierarchyEncoder(
        feature_dim=8,
        hidden=32,
        layers=1,
        heads=4,
        max_slots=8,
        min_slots=2,
        assignment_top_k=2,
    )


def test_sparse_hierarchy_is_permutation_equivariant() -> None:
    torch.manual_seed(3)
    model = _model().eval()
    features, b2b, p2b, pins, n_b2b, n_p2b, n_pins = _inputs()
    permutation = torch.tensor([3, 0, 5, 1, 6, 2, 4])
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(len(permutation))
    permuted_b2b = b2b[:, :, :2].long()
    permuted_b2b = inverse[permuted_b2b]
    permuted_b2b = torch.cat([permuted_b2b.float(), b2b[:, :, 2:]], dim=-1)
    permuted_p2b = p2b.clone()
    permuted_p2b[:, :, 1] = inverse[p2b[:, :, 1].long()].float()

    result = model(
        features,
        b2b=b2b,
        p2b=p2b,
        pins=pins,
        n_b2b=n_b2b,
        n_p2b=n_p2b,
        n_pins=n_pins,
    )
    permuted = model(
        features[:, permutation],
        b2b=permuted_b2b,
        p2b=permuted_p2b,
        pins=pins,
        n_b2b=n_b2b,
        n_p2b=n_p2b,
        n_pins=n_pins,
    )

    assert torch.allclose(result.tokens[:, permutation], permuted.tokens, atol=1e-5)
    assert torch.allclose(result.slot_tokens, permuted.slot_tokens, atol=1e-5)
    assert torch.allclose(result.global_state, permuted.global_state, atol=1e-5)


def test_router_budget_varies_and_assignments_are_sparse() -> None:
    torch.manual_seed(7)
    model = _model().eval()
    features, b2b, p2b, pins, n_b2b, n_p2b, n_pins = _inputs()
    result = model(
        features,
        b2b=b2b,
        p2b=p2b,
        pins=pins,
        n_b2b=n_b2b,
        n_p2b=n_p2b,
        n_pins=n_pins,
    )

    counts = result.diagnostics["active_slot_count"]
    assert counts[0] < counts[1]
    assert int((result.token_assignments != 0).sum(dim=-1).max()) <= 2
    assert torch.allclose(result.slot_weights.sum(dim=1), torch.ones(2))


def test_explicit_constraints_drive_the_same_native_slot_encoder() -> None:
    torch.manual_seed(9)
    model = _model().eval()
    features, b2b, p2b, pins, n_b2b, n_p2b, n_pins = _inputs()
    constraints = features[..., :5]
    graph = {
        "b2b": b2b,
        "p2b": p2b,
        "pins": pins,
        "n_b2b": n_b2b,
        "n_p2b": n_p2b,
        "n_pins": n_pins,
    }
    implicit = model(features, **graph)
    explicit = model(features, constraints=constraints, **graph)
    changed = constraints.clone()
    changed[..., 0] = 1.0 - changed[..., 0]
    changed_result = model(features, constraints=changed, **graph)

    assert torch.allclose(implicit.slot_tokens, explicit.slot_tokens)
    assert torch.allclose(
        implicit.diagnostics["constraint_density"],
        explicit.diagnostics["constraint_density"],
    )
    assert not torch.allclose(explicit.tokens, changed_result.tokens)


def test_inactive_slots_receive_no_gradient() -> None:
    torch.manual_seed(11)
    model = _model()
    features, b2b, p2b, pins, n_b2b, n_p2b, n_pins = _inputs()
    result = model(
        features,
        b2b=b2b,
        p2b=p2b,
        pins=pins,
        n_b2b=n_b2b,
        n_p2b=n_p2b,
        n_pins=n_pins,
    )
    inactive = ~result.active_slot_mask
    loss = (result.slot_tokens * inactive[:, :, None]).sum()
    loss.backward()

    gradient = model.router.slot_queries.grad
    assert gradient is not None
    assert torch.allclose(gradient, torch.zeros_like(gradient))


def test_graph_conditioned_sparse_flow_shapes_and_diagnostics() -> None:
    torch.manual_seed(13)
    model = ChipSparseHierarchyFlowModel(
        feature_dim=8,
        hidden=32,
        global_dim=32,
        target_layers=1,
        condition_layers=1,
        decoder_layers=1,
        heads=4,
        max_slots=4,
        min_slots=2,
        assignment_top_k=2,
        decoder_top_k=2,
    )
    features, b2b, p2b, pins, n_b2b, n_p2b, n_pins = _inputs()
    features = torch.cat([features, features[:, :2]], dim=1)
    target = torch.randn(2, 9, 3)
    graph = {
        "b2b": b2b,
        "p2b": p2b,
        "pins": pins,
        "n_b2b": n_b2b,
        "n_p2b": n_p2b,
        "n_pins": n_pins,
    }
    context = model.encode_condition_context(features, **graph)
    state = model.encode_target(target, features, **graph)
    decoded, diagnostics = model.decode_with_diagnostics(
        state,
        context,
        features,
    )
    sampled, sample_diagnostics = model.sample(
        features,
        sampling_steps=2,
        generator=torch.Generator().manual_seed(17),
        **graph,
    )

    assert state.shape == (2, 32)
    assert decoded.shape == (2, 9, 3)
    assert sampled.shape == decoded.shape
    assert torch.isfinite(decoded).all()
    assert diagnostics["active_slot_count"].shape == (2,)
    assert diagnostics["router_load"].shape == (2, 4)
    assert diagnostics["router_budgets"].shape == (2, 4)
    assert not torch.allclose(
        diagnostics["router_budget_per_case"],
        diagnostics["router_budget_per_case"][:1].expand(2),
    )
    assert torch.isfinite(diagnostics["router_load"]).all()
    assert float(sample_diagnostics["router_budget_min"]) >= 1.0


def test_sparse_flow_has_no_dense_token_attention() -> None:
    model = ChipSparseHierarchyFlowModel(
        feature_dim=8,
        hidden=32,
        global_dim=32,
        target_layers=1,
        condition_layers=1,
        decoder_layers=2,
        heads=4,
        max_slots=4,
        min_slots=2,
        assignment_top_k=2,
        decoder_top_k=2,
    )
    assert not any(isinstance(module, SparseAttention) for module in model.modules())


def test_parameter_matched_global_slot_control_is_within_two_percent() -> None:
    sparse = ChipSparseHierarchyFlowModel(
        feature_dim=8,
        hidden=32,
        global_dim=32,
        target_layers=1,
        condition_layers=1,
        decoder_layers=1,
        heads=4,
        max_slots=4,
        min_slots=2,
        assignment_top_k=2,
        decoder_top_k=2,
    )
    config, parameters, relative_error = parameter_matched_global_slot_config(
        {
            "feature_dim": 8,
            "hidden": 32,
            "global_dim": 32,
            "target_layers": 1,
            "condition_layers": 1,
            "decoder_layers": 1,
            "heads": 4,
        },
        count_parameters(sparse),
    )

    assert config["architecture"] == "global_slot"
    assert parameters > 0
    assert relative_error <= 0.02


def test_straight_through_slots_train_budget_and_active_queries() -> None:
    torch.manual_seed(15)
    model = ChipSparseHierarchyFlowModel(
        feature_dim=8,
        hidden=32,
        global_dim=32,
        target_layers=1,
        condition_layers=1,
        decoder_layers=1,
        heads=4,
        max_slots=4,
        min_slots=2,
        assignment_top_k=2,
        assignment_temperature=0.5,
        decoder_top_k=2,
    )
    features, b2b, p2b, pins, n_b2b, n_p2b, n_pins = _inputs()
    graph = {
        "b2b": b2b,
        "p2b": p2b,
        "pins": pins,
        "n_b2b": n_b2b,
        "n_p2b": n_p2b,
        "n_pins": n_pins,
    }
    context = model.encode_condition_context(features, **graph)
    decoded, decode_diagnostics = model.decode_with_diagnostics(
        context.global_state,
        context,
        features,
    )
    regularization = model.sparse_regularization(
        context,
        decode_diagnostics,
    )
    loss = (
        decoded.square().mean()
        + regularization["budget"]
        + regularization["route_balance_total"]
    )
    loss.backward()

    query_gradient = model.condition_hierarchy.router.slot_queries.grad
    budget_gradient = model.condition_hierarchy.router.budget_head.weight.grad
    assert query_gradient is not None
    assert budget_gradient is not None
    active = context.hierarchy.active_slot_mask.any(dim=0)
    assert torch.isfinite(query_gradient).all()
    assert float(query_gradient[active].abs().sum()) > 0.0
    assert float(budget_gradient.abs().sum()) > 0.0
    assert torch.isfinite(regularization["dead_slot_fraction"])
