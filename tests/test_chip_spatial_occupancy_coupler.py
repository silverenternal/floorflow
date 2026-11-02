from __future__ import annotations

import torch

from floorflow.chip_flow.checkpoint_loading import load_chip_checkpoint
from floorflow.chip_flow.hierarchical_flow import (
    HierarchicalTokenFlowModel,
    build_hierarchy_targets,
)
from floorflow.chip_flow.sparse_layers import prepare_state_edge_cache
from floorflow.chip_flow.spatial_occupancy_coupler import (
    SpatialOccupancyCoupler,
    block_geometry_in_chart,
    route_obstruction_by_block,
)


def _batch() -> dict:
    count = 8
    constraints = torch.zeros(2, count, 5)
    constraints[:, :, 3] = torch.tensor([1, 1, 2, 2, 3, 3, 4, 4])
    constraints[:, :2, 4] = 1
    return {
        "areas": torch.ones(2, count),
        "boxes": torch.stack(
            [
                torch.stack(
                    [
                        torch.arange(count) * 1.2,
                        torch.zeros(count),
                        torch.ones(count),
                        torch.ones(count),
                    ],
                    dim=-1,
                )
            ]
            * 2
        ),
        "constraints": constraints,
        "features": torch.zeros(2, count, 10),
        "target_latent": torch.randn(2, count, 3),
        "b2b": torch.tensor(
            [
                [[0.0, 1.0, 1.0], [1.0, 2.0, 1.0], [2.0, 3.0, 0.5]],
                [[0.0, 2.0, 1.0], [2.0, 4.0, 1.0], [4.0, 6.0, 0.5]],
            ]
        ),
        "p2b": torch.tensor(
            [
                [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [2.0, 3.0, 0.5]],
                [[0.0, 2.0, 1.0], [1.0, 4.0, 1.0], [2.0, 6.0, 0.5]],
            ]
        ),
        "pins": torch.tensor(
            [
                [[0.2, 0.4], [0.8, 0.6], [0.5, 0.5]],
                [[0.1, 0.5], [0.7, 0.9], [0.4, 0.3]],
            ]
        ),
        "n_b2b": torch.tensor([3, 3]),
        "n_p2b": torch.tensor([3, 3]),
        "n_pins": torch.tensor([3, 3]),
    }


def _model(spatial: bool) -> HierarchicalTokenFlowModel:
    return HierarchicalTokenFlowModel(
        hidden=64,
        heads=4,
        depth=2,
        typed_layers=1,
        head_dim=64,
        spatial_occupancy_conditioning=spatial,
    )


def test_spatial_occupancy_is_zero_gated_and_trainable() -> None:
    torch.manual_seed(3)
    module = SpatialOccupancyCoupler(64)
    arguments = (
        torch.randn(2, 8, 64),
        torch.randn(2, 8, 3),
        torch.rand(2, 8) + 0.1,
    )
    residual, diagnostics = module(*arguments)
    assert torch.equal(residual, torch.zeros_like(residual))
    assert diagnostics["spatial_occupancy_grid_fill"] > 0
    module.gate.data.fill_(1.0)
    active, _ = module(*arguments)
    assert active.abs().sum() > 0
    active.square().mean().backward()
    assert module.reduce[-1].weight.grad is not None
    assert module.reduce[-1].weight.grad.abs().sum() > 0


def test_spatial_occupancy_uses_normalized_centers_without_sigmoid() -> None:
    state = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
            ]
        ]
    )

    centers, _extent, _lower, _span = block_geometry_in_chart(
        state,
        torch.ones(1, 2),
    )

    assert centers[0, 0, 0] < 0.25
    assert centers[0, 1, 0] > 0.75


def test_block_geometry_uses_the_same_pin_chart_as_route_fields() -> None:
    areas = torch.full((1, 2), 2.0)
    state = torch.tensor(
        [
            [
                [0.5, 0.0, 0.0],
                [0.5, 0.0, 0.0],
            ]
        ]
    )
    lower = torch.tensor([[-2.0, -2.0]])
    span = torch.tensor([[4.0]])

    centers, extent, chart_lower, chart_span = block_geometry_in_chart(
        state,
        areas,
        lower=lower,
        span=span,
    )

    torch.testing.assert_close(centers[0, 0], torch.tensor([0.75, 0.5]))
    torch.testing.assert_close(
        extent[0, 0],
        torch.tensor([2.0**0.5 / 4.0, 2.0**0.5 / 4.0]),
    )
    torch.testing.assert_close(chart_lower, lower)
    torch.testing.assert_close(chart_span, span)


def test_route_aware_spatial_field_is_scale_equivariant() -> None:
    torch.manual_seed(23)
    module = SpatialOccupancyCoupler(16, route_aware=True)
    module.gate.data.fill_(1.0)
    torch.nn.init.normal_(module.obstruction_encoder.weight, std=0.02)
    tokens = torch.randn(1, 4, 16)
    state = torch.randn(1, 4, 3)
    areas = torch.ones(1, 4)
    b2b = torch.tensor([[[0.0, 1.0, 1.0], [2.0, 3.0, 0.5]]])
    p2b = torch.tensor([[[0.0, 0.0, 1.0], [1.0, 2.0, 0.5]]])
    pins = torch.tensor([[[0.2, 0.3], [0.6, 0.8]]])

    def cache(scale: float) -> dict:
        return prepare_state_edge_cache(
            areas=areas * scale**2,
            b2b=b2b,
            p2b=p2b,
            pins=pins * scale,
            n_b2b=torch.tensor([2]),
            n_p2b=torch.tensor([2]),
            n_pins=torch.tensor([2]),
        )

    original, _ = module(tokens, state, areas, cache=cache(1.0))
    scaled, _ = module(tokens, state, areas * 100.0, cache=cache(10.0))

    torch.testing.assert_close(original, scaled, atol=2.0e-5, rtol=2.0e-5)


def test_route_channels_are_normalized_independently() -> None:
    module = SpatialOccupancyCoupler(16, route_aware=True, grid_size=4)
    route = module._route_fields(
        torch.tensor([[[0.1, 0.1], [0.9, 0.9]]]),
        torch.ones(1, 2),
        {
            "b2b_source": torch.tensor([0]),
            "b2b_target": torch.tensor([1]),
            "b2b_weight": torch.tensor([100.0]),
            "p2b_pin": torch.tensor([0]),
            "p2b_block": torch.tensor([1]),
            "p2b_weight": torch.tensor([0.1]),
            "pins": torch.tensor([[0.1, 0.9]]),
        },
        lower=torch.zeros(1, 2),
        span=torch.ones(1, 1),
        grid=4,
        dtype=torch.float32,
    )

    torch.testing.assert_close(route[0, 0].max(), torch.tensor(1.0))
    torch.testing.assert_close(route[0, 1].max(), torch.tensor(1.0))


def test_route_obstruction_responds_to_macro_blocking_an_l_path() -> None:
    grid = 8
    centers = torch.tensor([[[0.1, 0.5], [0.5, 0.5], [0.9, 0.5]]])
    cache = prepare_state_edge_cache(
        areas=torch.ones(1, 3),
        b2b=torch.tensor([[[0.0, 2.0, 1.0]]]),
        p2b=torch.empty(1, 0, 3),
        pins=torch.empty(1, 0, 2),
        n_b2b=torch.tensor([1]),
        n_p2b=torch.tensor([0]),
        n_pins=torch.tensor([0]),
    )
    clear = torch.zeros(1, 5, grid * grid)
    blocked = clear.clone()
    midpoint = (grid // 2) * grid + (grid // 2)
    blocked[:, 1, midpoint] = 1.0

    clear_features = route_obstruction_by_block(
        clear,
        centers,
        cache,
        lower=torch.zeros(1, 2),
        span=torch.ones(1, 1),
        grid=grid,
    )
    blocked_features = route_obstruction_by_block(
        blocked,
        centers,
        cache,
        lower=torch.zeros(1, 2),
        span=torch.ones(1, 1),
        grid=grid,
    )

    assert blocked_features[0, 0, 0] > clear_features[0, 0, 0]
    assert blocked_features[0, 2, 0] > clear_features[0, 2, 0]
    assert torch.equal(blocked_features[0, 1], torch.zeros(2))


def test_route_obstruction_reaches_block_state_gradient() -> None:
    torch.manual_seed(29)
    module = SpatialOccupancyCoupler(16, route_aware=True)
    module.gate.data.fill_(1.0)
    torch.nn.init.normal_(module.obstruction_encoder.weight, std=0.05)
    tokens = torch.randn(1, 4, 16)
    state = torch.randn(1, 4, 3, requires_grad=True)
    areas = torch.ones(1, 4)
    cache = prepare_state_edge_cache(
        areas=areas,
        b2b=torch.tensor([[[0.0, 1.0, 1.0], [2.0, 3.0, 0.5]]]),
        p2b=torch.tensor([[[0.0, 0.0, 1.0]]]),
        pins=torch.tensor([[[0.2, 0.3]]]),
        n_b2b=torch.tensor([2]),
        n_p2b=torch.tensor([1]),
        n_pins=torch.tensor([1]),
    )

    residual, _ = module(tokens, state, areas, cache=cache)
    residual.square().mean().backward()

    assert state.grad is not None
    assert torch.isfinite(state.grad).all()
    assert state.grad.abs().sum() > 0


def test_spatial_occupancy_is_block_permutation_equivariant() -> None:
    torch.manual_seed(5)
    module = SpatialOccupancyCoupler(64)
    module.gate.data.fill_(1.0)
    tokens = torch.randn(1, 8, 64)
    state = torch.randn(1, 8, 3)
    areas = torch.rand(1, 8) + 0.1
    original, _ = module(tokens, state, areas)
    permutation = torch.tensor([3, 0, 7, 2, 5, 1, 6, 4])
    reordered, _ = module(
        tokens[:, permutation],
        state[:, permutation],
        areas[:, permutation],
    )
    torch.testing.assert_close(
        original[:, permutation],
        reordered,
        atol=2.0e-5,
        rtol=2.0e-5,
    )


def test_spatial_occupancy_zero_gate_preserves_base_prediction() -> None:
    torch.manual_seed(7)
    base = _model(False)
    torch.manual_seed(7)
    coupled = _model(True)
    coupled.load_state_dict(base.state_dict(), strict=False)
    batch = _batch()
    base_context = base.prepare_context(batch)
    coupled_context = coupled.prepare_context(batch)
    target = build_hierarchy_targets(batch, base_context)
    time = torch.full((2,), 0.5)
    torch.testing.assert_close(
        base(target, time, base_context).block_state,
        coupled(target, time, coupled_context).block_state,
    )
    assert coupled.spatial_occupancy is not None


def test_spatial_occupancy_checkpoint_round_trip() -> None:
    model = _model(True)
    checkpoint = {
        "config": {
            "architecture": "typed_hierarchy",
            "hidden": 64,
            "heads": 4,
            "depth": 2,
            "head_dim": 64,
            "typed_layers": 1,
            "spatial_occupancy_conditioning": True,
        },
        "model": model.state_dict(),
        "ema": model.state_dict(),
    }
    restored = load_chip_checkpoint(checkpoint, "cpu", use_ema=True)
    assert isinstance(restored, HierarchicalTokenFlowModel)
    assert restored.spatial_occupancy is not None


def test_old_route_field_checkpoint_adds_zero_obstruction_encoder() -> None:
    model = _model(True)
    model.spatial_occupancy = SpatialOccupancyCoupler(
        64,
        route_aware=True,
    )
    legacy = {
        key: value
        for key, value in model.state_dict().items()
        if not key.startswith("spatial_occupancy.obstruction_encoder.")
    }
    checkpoint = {
        "config": {
            "architecture": "typed_hierarchy",
            "hidden": 64,
            "heads": 4,
            "depth": 2,
            "head_dim": 64,
            "typed_layers": 1,
            "spatial_occupancy_conditioning": True,
            "spatial_route_field": True,
        },
        "model": legacy,
    }

    restored = load_chip_checkpoint(checkpoint, "cpu", use_ema=False)

    encoder = restored.spatial_occupancy.obstruction_encoder
    assert encoder is not None
    assert torch.count_nonzero(encoder.weight) == 0
    assert torch.count_nonzero(encoder.bias) == 0


def test_spatial_occupancy_parameter_budget_is_under_one_percent() -> None:
    module = SpatialOccupancyCoupler(192)
    base_parameters = 5_836_447
    added = sum(parameter.numel() for parameter in module.parameters())
    assert added < base_parameters * 0.01
