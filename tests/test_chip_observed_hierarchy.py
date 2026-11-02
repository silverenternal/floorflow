from __future__ import annotations

import torch

from floorflow.chip_flow.checkpoint_loading import load_chip_checkpoint
from floorflow.chip_flow.hierarchical_flow import (
    HierarchicalTokenFlowModel,
    build_hierarchy_targets,
)
from floorflow.chip_flow.hierarchy_sampling import sample_hierarchy_flow
from floorflow.chip_flow.observed_hierarchy import (
    observed_hierarchy_state,
)
from floorflow.chip_flow.shape_mode_head import attach_shape_mode_head


def _batch() -> dict:
    count = 4
    constraints = torch.zeros(1, count, 5)
    constraints[..., 3] = torch.tensor([1.0, 1.0, 2.0, 2.0])
    return {
        "areas": torch.ones(1, count),
        "boxes": torch.tensor(
            [[[0.0, 0.0, 1.0, 1.0], [1.0, 0.0, 1.0, 1.0],
              [2.5, 0.0, 1.0, 1.0], [3.5, 0.0, 1.0, 1.0]]]
        ),
        "constraints": constraints,
        "features": torch.zeros(1, count, 10),
        "target_latent": torch.randn(1, count, 3),
        "b2b": torch.tensor([[[0.0, 1.0, 1.0], [2.0, 3.0, 1.0]]]),
        "p2b": torch.tensor([[[0.0, 0.0, 1.0], [1.0, 2.0, 1.0]]]),
        "pins": torch.tensor([[[0.2, 0.3], [2.2, 0.3]]]),
        "n_b2b": torch.tensor([2]),
        "n_p2b": torch.tensor([2]),
        "n_pins": torch.tensor([2]),
    }


def _model(*, observed: bool) -> HierarchicalTokenFlowModel:
    return HierarchicalTokenFlowModel(
        hidden=32,
        heads=4,
        depth=1,
        head_dim=16,
        typed_layers=1,
        observed_hierarchy_coupling=observed,
    )


def test_current_block_observations_have_hierarchy_shapes() -> None:
    block_state = torch.tensor(
        [[[0.2, 0.3, 0.0], [0.7, 0.3, 0.0],
          [1.2, 0.3, 0.0], [1.7, 0.3, 0.0]]]
    )
    membership = torch.tensor(
        [[[True, False, False, False],
          [True, False, False, False],
          [False, True, False, False],
          [False, True, False, False]]]
    )
    incidence = torch.tensor(
        [[[True, False], [False, False], [False, True], [False, False]]]
    )

    observed = observed_hierarchy_state(
        block_state,
        torch.ones(1, 4),
        membership,
        torch.tensor([[True, True, False, False]]),
        incidence,
        torch.tensor([[True, True]]),
    )

    assert observed.global_state.shape == (1, 4)
    assert observed.structural_state.shape == (1, 4, 3)
    assert observed.port_center.shape == (1, 2, 2)
    assert torch.isfinite(observed.global_state).all()
    torch.testing.assert_close(
        observed.port_center[0, 0],
        block_state[0, 0, :2],
    )
    torch.testing.assert_close(
        observed.port_center[0, 1],
        block_state[0, 2, :2],
    )


def test_observed_coupling_is_identity_when_adapters_are_zero() -> None:
    torch.manual_seed(7)
    base = _model(observed=False)
    coupled = _model(observed=True)
    coupled.load_state_dict(base.state_dict(), strict=False)
    batch = _batch()
    base_context = base.prepare_context(batch)
    coupled_context = coupled.prepare_context(batch)
    target = build_hierarchy_targets(batch, base_context)
    time = torch.full((1,), 0.5)

    expected = base(target, time, base_context)
    actual = coupled(target, time, coupled_context)

    torch.testing.assert_close(actual.block_state, expected.block_state)
    assert coupled.observed_hierarchy_coupler is not None


def test_observed_coupling_changes_state_and_reaches_adapter_gradients() -> None:
    torch.manual_seed(11)
    base = _model(observed=False)
    coupled = _model(observed=True)
    coupled.load_state_dict(base.state_dict(), strict=False)
    coupler = coupled.observed_hierarchy_coupler
    assert coupler is not None
    with torch.no_grad():
        for adapter in (
            coupler.global_adapter,
            coupler.structural_adapter,
            coupler.port_adapter,
        ):
            adapter.weight.normal_(0.0, 0.1)
    batch = _batch()
    context = coupled.prepare_context(batch)
    target = build_hierarchy_targets(batch, context)

    output = coupled(target, torch.full((1,), 0.5), context)
    loss = (
        output.global_state.square().mean()
        + output.structural_state.square().mean()
        + output.port_state.square().mean()
        + output.block_state.square().mean()
    )
    loss.backward()

    assert coupler.global_adapter.weight.grad is not None
    assert coupler.global_adapter.weight.grad.abs().sum() > 0.0
    assert coupler.structural_adapter.weight.grad is not None
    assert coupler.port_adapter.weight.grad is not None


def test_checkpoint_loader_restores_observed_coupling() -> None:
    model = _model(observed=True)
    checkpoint = {
        "config": {
            "architecture": "typed_hierarchy",
            "hidden": 32,
            "heads": 4,
            "depth": 1,
            "head_dim": 16,
            "typed_layers": 1,
            "observed_hierarchy_coupling": True,
        },
        "model": model.state_dict(),
    }

    restored = load_chip_checkpoint(checkpoint, "cpu", use_ema=False)

    assert restored.observed_hierarchy_coupler is not None


def test_corrected_hierarchy_branches_run_together() -> None:
    torch.manual_seed(19)
    model = HierarchicalTokenFlowModel(
        hidden=32,
        heads=4,
        depth=1,
        head_dim=16,
        typed_layers=1,
        relation_specific=True,
        observed_hierarchy_coupling=True,
        port_consensus_conditioning=True,
        spatial_route_field=True,
    )
    attach_shape_mode_head(model, modes=2, width=8, apply=True)
    batch = _batch()
    batch["source_latent"] = torch.randn(1, 4, 3)

    generated, diagnostics = sample_hierarchy_flow(
        model,
        batch,
        sampling_steps=2,
        generator=torch.Generator().manual_seed(7),
        projector=False,
        projector_iterations=2,
    )

    assert torch.isfinite(generated).all()
    assert diagnostics["shape_mode_logits"].shape == (1, 2)


def test_pareto_baseline_skips_extra_shape_projection(monkeypatch) -> None:
    torch.manual_seed(31)
    base = HierarchicalTokenFlowModel(
        hidden=32,
        heads=4,
        depth=1,
        head_dim=16,
        typed_layers=1,
        projector=True,
        projector_iterations=1,
        projector_grid_size=4,
    )
    candidate = HierarchicalTokenFlowModel(
        hidden=32,
        heads=4,
        depth=1,
        head_dim=16,
        typed_layers=1,
        projector=True,
        projector_iterations=1,
        projector_grid_size=4,
    )
    candidate.load_state_dict(base.state_dict(), strict=False)
    attach_shape_mode_head(
        candidate,
        modes=2,
        width=8,
        apply=True,
        selection="pareto",
    )
    monkeypatch.setattr(
        "floorflow.chip_flow.hierarchy_sampling."
        "select_pareto_safe_shape_mode",
        lambda features: torch.zeros(
            features.shape[0],
            dtype=torch.long,
            device=features.device,
        ),
    )
    batch = _batch()
    batch["source_latent"] = torch.randn(1, 4, 3)

    expected, _ = sample_hierarchy_flow(
        base,
        batch,
        sampling_steps=1,
        generator=torch.Generator().manual_seed(9),
        projector=True,
        projector_iterations=1,
        projector_grid_size=4,
    )
    actual, _ = sample_hierarchy_flow(
        candidate,
        batch,
        sampling_steps=1,
        generator=torch.Generator().manual_seed(9),
        projector=True,
        projector_iterations=1,
        projector_grid_size=4,
    )

    torch.testing.assert_close(actual, expected)
