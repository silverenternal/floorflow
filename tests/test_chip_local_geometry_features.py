from __future__ import annotations

import torch

from floorflow.chip_flow.checkpoint_loading import load_chip_checkpoint
from floorflow.chip_flow.hierarchical_flow import HierarchicalTokenFlowModel
from floorflow.chip_flow.local_geometry_features import (
    LOCAL_GEOMETRY_DIM,
    RawLocalGeometryEncoder,
    raw_local_geometry_features,
)


def _raw_inputs() -> tuple:
    areas = torch.ones(1, 3)
    b2b = torch.tensor([[[0.0, 1.0, 0.5], [0.0, 2.0, 2.0]]])
    p2b = torch.tensor([[[0.0, 0.0, 1.0], [1.0, 0.0, 0.5]]])
    pins = torch.tensor([[[0.1, 0.2], [0.7, 0.9]]])
    return (
        areas,
        b2b,
        p2b,
        pins,
        torch.tensor([2]),
        torch.tensor([2]),
        torch.tensor([2]),
    )


def test_raw_local_geometry_features_capture_edge_and_pin_distributions() -> None:
    features = raw_local_geometry_features(*_raw_inputs())
    assert features.shape == (1, 3, LOCAL_GEOMETRY_DIM)
    assert torch.isfinite(features).all()
    assert features[0, 0, 0] > features[0, 1, 0]
    assert features[0, 0, 3] > 0
    assert features[0, 0, 7] >= 0


def test_raw_local_geometry_pin_spread_is_scale_equivariant() -> None:
    areas, b2b, p2b, pins, n_b2b, n_p2b, n_pins = _raw_inputs()

    original = raw_local_geometry_features(
        areas,
        b2b,
        p2b,
        pins,
        n_b2b,
        n_p2b,
        n_pins,
    )
    scaled = raw_local_geometry_features(
        areas * 100.0,
        b2b,
        p2b,
        pins * 10.0,
        n_b2b,
        n_p2b,
        n_pins,
    )

    torch.testing.assert_close(original, scaled, atol=1.0e-5, rtol=1.0e-5)


def test_raw_local_geometry_encoder_is_zero_gated_and_trainable() -> None:
    encoder = RawLocalGeometryEncoder(16)
    features = torch.randn(2, 4, LOCAL_GEOMETRY_DIM)
    initial = encoder(features)
    assert torch.equal(initial, torch.zeros(2, 4, 16))
    initial.sum().backward()
    assert encoder.gate.grad is not None and encoder.gate.grad.abs().sum() > 0
    encoder.zero_grad(set_to_none=True)
    encoder.gate.data.fill_(1.0)
    output = encoder(features)
    assert output.abs().sum() > 0
    output.square().mean().backward()
    assert encoder.network[-1].weight.grad is not None


def test_raw_local_geometry_handles_all_empty_edges() -> None:
    features = raw_local_geometry_features(
        torch.ones(2, 4),
        torch.empty(2, 0, 3),
        torch.empty(2, 0, 3),
        torch.empty(2, 0, 2),
        torch.zeros(2, dtype=torch.long),
        torch.zeros(2, dtype=torch.long),
        torch.zeros(2, dtype=torch.long),
    )
    assert features.shape == (2, 4, LOCAL_GEOMETRY_DIM)
    assert torch.equal(features, torch.zeros_like(features))


def test_local_geometry_branch_is_backward_compatible_with_old_checkpoints() -> None:
    model = HierarchicalTokenFlowModel(
        hidden=16,
        heads=2,
        depth=1,
        head_dim=16,
        typed_layers=1,
    )
    old_state = {
        name: value
        for name, value in model.state_dict().items()
        if not name.startswith("typed.local_geometry_encoder.")
    }
    restored = load_chip_checkpoint(
        {
            "config": {
                "architecture": "typed_hierarchy",
                "hidden": 16,
                "heads": 2,
                "depth": 1,
                "head_dim": 16,
                "typed_layers": 1,
            },
            "ema": old_state,
            "model": old_state,
        },
        "cpu",
        use_ema=True,
    )
    assert restored.typed.local_geometry_encoder.gate.item() == 0.0
