from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from floorflow.chip_flow.hierarchical_flow import (
    HierarchicalTokenFlowModel,
    build_hierarchy_targets,
)
from floorflow.chip_flow.hierarchy_state import HierarchyPrediction
from floorflow.chip_flow.nfe_consistency import hierarchy_nfe_consistency_loss
from floorflow.chip_flow.training_validation import validate_training_scope


def _batch() -> dict:
    count = 4
    constraints = torch.zeros(1, count, 5)
    constraints[0, :2, 3] = 1
    constraints[0, 2:, 3] = 2
    return {
        "areas": torch.ones(1, count),
        "boxes": torch.tensor(
            [[[0.0, 0.0, 1.0, 1.0], [1.0, 0.0, 1.0, 1.0],
              [4.0, 0.0, 1.0, 1.0], [5.0, 0.0, 1.0, 1.0]]]
        ),
        "constraints": constraints,
        "features": torch.zeros(1, count, 10),
        "b2b": torch.empty(1, 0, 3),
        "p2b": torch.empty(1, 0, 3),
        "pins": torch.empty(1, 0, 2),
        "n_b2b": torch.zeros(1, dtype=torch.long),
        "n_p2b": torch.zeros(1, dtype=torch.long),
        "n_pins": torch.zeros(1, dtype=torch.long),
        "target_latent": torch.randn(1, count, 3),
    }


def test_nfe_consistency_is_zero_when_disabled_and_reaches_model_gradient() -> None:
    torch.manual_seed(41)
    model = HierarchicalTokenFlowModel(
        hidden=32,
        heads=4,
        depth=1,
        head_dim=16,
        typed_layers=1,
    )
    batch = _batch()
    context = model.prepare_context(batch)
    target = build_hierarchy_targets(batch, context)
    source = type(target)(
        global_state=target.global_state + 0.1,
        structural_state=target.structural_state + 0.1,
        block_state=target.block_state + 0.1,
        port_state=target.port_state,
    )
    time = torch.full((1,), 0.25)
    prediction = model(source, time, context)
    disabled = SimpleNamespace(nfe_consistency_weight=0.0)
    loss, terms = hierarchy_nfe_consistency_loss(
        model,
        source,
        time,
        prediction,
        target,
        context,
        disabled,
        step=0,
    )
    assert loss == 0
    assert terms == {}

    enabled = SimpleNamespace(
        nfe_consistency_weight=0.1,
        nfe_consistency_steps=1,
        nfe_consistency_target_weight=1.0,
        nfe_consistency_self_weight=0.1,
        flow_huber_beta=0.1,
    )
    loss, terms = hierarchy_nfe_consistency_loss(
        model,
        source,
        time,
        prediction,
        target,
        context,
        enabled,
        step=1,
    )
    assert loss > 0
    assert torch.isfinite(loss)
    loss.backward()
    assert model.hierarchy_to_latent.weight.grad is not None
    assert terms["nfe_consistency_endpoint"] > 0


def test_nfe_consistency_preserves_native_group_moment_state() -> None:
    torch.manual_seed(43)
    model = HierarchicalTokenFlowModel(
        hidden=32,
        heads=4,
        depth=1,
        head_dim=16,
        typed_layers=1,
        structural_relations=True,
        group_moment_conditioning=True,
    )
    batch = _batch()
    context = model.prepare_context(batch)
    target = build_hierarchy_targets(batch, context)
    assert target.structural_moment_state is not None
    source = replace(
        target,
        global_state=target.global_state + 0.1,
        structural_state=target.structural_state + 0.1,
        block_state=target.block_state + 0.1,
    )
    time = torch.full((1,), 0.25)
    prediction = model(source, time, context)
    args = SimpleNamespace(
        nfe_consistency_weight=0.1,
        nfe_consistency_steps=1,
        nfe_consistency_target_weight=1.0,
        nfe_consistency_self_weight=0.1,
        flow_huber_beta=0.1,
    )

    loss, _terms = hierarchy_nfe_consistency_loss(
        model,
        source,
        time,
        prediction,
        target,
        context,
        args,
        step=1,
    )

    assert torch.isfinite(loss)
    loss.backward()
    assert model.group_moment_coupler.native_residual[-1].weight.grad is not None


def test_nfe_consistency_advances_native_group_moment_state() -> None:
    torch.manual_seed(47)
    context_model = HierarchicalTokenFlowModel(
        hidden=32,
        heads=4,
        depth=1,
        head_dim=16,
        typed_layers=1,
        structural_relations=True,
        group_moment_conditioning=True,
    )
    batch = _batch()
    context = context_model.prepare_context(batch)
    target = build_hierarchy_targets(batch, context)
    assert target.structural_moment_state is not None
    source = replace(
        target,
        structural_moment_state=torch.zeros_like(
            target.structural_moment_state
        ),
    )
    captured = []

    class ProbeModel(torch.nn.Module):
        def forward(self, state, time, _context, **_kwargs):
            del time
            captured.append(state.structural_moment_state.clone())
            return HierarchyPrediction(
                global_state=target.global_state,
                structural_state=target.structural_state,
                block_state=target.block_state,
                port_state=target.port_state,
                structural_moment_state=torch.ones_like(
                    target.structural_moment_state
                ),
            )

    args = SimpleNamespace(
        nfe_consistency_weight=0.1,
        nfe_consistency_steps=2,
        nfe_consistency_target_weight=1.0,
        nfe_consistency_self_weight=0.1,
        flow_huber_beta=0.1,
    )
    loss, _terms = hierarchy_nfe_consistency_loss(
        ProbeModel(),
        source,
        torch.full((1,), 0.25),
        HierarchyPrediction(
            global_state=target.global_state,
            structural_state=target.structural_state,
            block_state=target.block_state,
            port_state=target.port_state,
            structural_moment_state=torch.ones_like(
                target.structural_moment_state
            ),
        ),
        target,
        context,
        args,
        step=1,
    )

    assert torch.isfinite(loss)
    assert len(captured) == 2
    assert captured[0].mean() > 0.0
    assert captured[1].mean() > captured[0].mean()


def test_deployment_source_gate_rejects_random_training_source() -> None:
    args = SimpleNamespace(
        train_indices=object(),
        allow_full_data=False,
        batch_size=2,
        grad_accum_steps=1,
        hierarchy_source_mode="random",
        hierarchy_source_levels="random",
        require_deployment_source=True,
        hierarchy_source_noise=0.0,
        hierarchy_source_shelf_probability=0.5,
        hierarchy_source_shelf_probability_final=None,
        hierarchy_source_shelf_probability_steps=0,
        backbone_unfreeze_step=None,
        structured_legalizer_apply=False,
        endpoint_legalizer_target_weight=0.0,
        coarse_commit_fraction=None,
        relation_specific_attention=False,
        typed_mode="typed",
        structural_relation_head=False,
        structured_heads=False,
        geometry_augment_probability=1.0,
        backbone_lr_scale=1.0,
        structured_head_lr_scale=1.0,
        legalizer_head_lr_scale=1.0,
        structured_head_warmup_steps=0,
        legalizer_head_warmup_steps=0,
        grad_clip=1.0,
        ema_decay=0.9995,
        ema_warmup_steps=0,
        checkpoint_policy="final",
        checkpoint_shortlist_size=0,
        hierarchy_forced_head_weight=0.0,
        endpoint_compatibility_iterations=2,
        training_projector_iterations=2,
        training_projector_strength=1.0,
        structured_legalizer_strength=1.0,
        structured_legalizer_max_displacement=0.1,
        projector_grid_size=16,
        endpoint_refiner_trajectory_steps=2,
        nfe_consistency_steps=1,
        endpoint_pin_tail_fraction=0.2,
    )
    with pytest.raises(ValueError, match="deployment-source"):
        validate_training_scope(args)
