"""Tests for multi-moment group conditioning."""

from __future__ import annotations

from dataclasses import replace

import torch

from floorflow.chip_flow.checkpoint_loading import load_chip_checkpoint
from floorflow.chip_flow.group_moment_coupler import (
    GroupMomentCoupler,
    apply_native_group_moment,
)
from floorflow.chip_flow.hierarchical_advance import advance_hierarchy_state
from floorflow.chip_flow.hierarchical_flow import (
    HierarchicalTokenFlowModel,
    build_hierarchy_targets,
)
from floorflow.chip_flow.hierarchical_flow_runtime import (
    hierarchy_flow_loss,
    interpolate_hierarchy,
)
from floorflow.chip_flow.hierarchy_state import HierarchyPrediction, HierarchyState
from floorflow.chip_flow.training_manifest import (
    RESUME_CRITICAL_KEYS,
    RESUME_NEW_PARAM_PREFIXES,
)


def _batch() -> dict:
    count = 7
    constraints = torch.zeros(2, count, 5)
    constraints[:, :, 4] = torch.tensor(
        [[1.0, 0.0, 2.0, 0.0, 0.0, 4.0, 0.0]] * 2
    )
    constraints[:, :2, 2] = 1
    constraints[0, :2, 3] = 1
    constraints[0, 2:5, 3] = 2
    constraints[1, :3, 3] = 1
    constraints[1, 3:6, 3] = 3
    boxes = torch.stack(
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
    )
    return {
        "areas": torch.ones(2, count),
        "boxes": boxes,
        "constraints": constraints,
        "features": torch.zeros(2, count, 10),
        "target_latent": torch.randn(2, count, 3),
        "b2b": torch.tensor(
            [
                [[0.0, 1.0, 1.0], [1.0, 2.0, 0.5]],
                [[0.0, 2.0, 1.0], [2.0, 4.0, 0.5]],
            ]
        ),
        "p2b": torch.tensor(
            [
                [[0.0, 0.0, 1.0], [1.0, 1.0, 0.5]],
                [[0.0, 2.0, 1.0], [1.0, 4.0, 0.5]],
            ]
        ),
        "pins": torch.tensor(
            [
                [[0.2, 0.4], [0.8, 0.6], [0.0, 0.0]],
                [[0.1, 0.5], [0.7, 0.9], [0.0, 0.0]],
            ]
        ),
        "n_b2b": torch.tensor([2, 2]),
        "n_p2b": torch.tensor([2, 2]),
        "n_pins": torch.tensor([2, 2]),
    }


def _model(*, group_moments: bool) -> HierarchicalTokenFlowModel:
    return HierarchicalTokenFlowModel(
        hidden=24,
        heads=3,
        depth=1,
        head_dim=8,
        typed_layers=1,
        structural_relations=group_moments,
        group_moment_conditioning=group_moments,
    )


def test_group_moment_coupler_is_zero_output_at_init_and_gradient_reachable() -> None:
    torch.manual_seed(41)
    coupler = GroupMomentCoupler(24, width=8)
    tokens = torch.randn(2, 7, 24)
    state = torch.randn(2, 7, 3) * 0.1
    moments = torch.randn(2, 9, 7)
    membership = torch.zeros(2, 7, 9, dtype=torch.bool)
    membership[:, :2, 0] = True
    membership[:, 2:5, 1] = True
    membership[:, 5:, 2] = True
    mask = membership.any(dim=1)
    zero = coupler(tokens, state, moments, membership, mask, torch.ones(2))
    assert torch.equal(zero, torch.zeros_like(zero))

    zero.sum().backward()
    assert coupler.residual[-1].weight.grad is not None
    assert coupler.residual[-1].weight.grad.abs().sum() > 0.0

    coupler.zero_grad(set_to_none=True)
    with torch.no_grad():
        torch.nn.init.normal_(coupler.residual[-1].weight, std=0.02)
    output = coupler(tokens, state, moments, membership, mask, torch.ones(2))
    assert output.abs().sum() > 0.0
    output.square().mean().backward()
    assert coupler.residual[-1].weight.grad is not None
    assert coupler.gate.grad is not None


def test_group_moment_coupler_preserves_slot_semantics() -> None:
    torch.manual_seed(47)
    coupler = GroupMomentCoupler(24, width=8)
    tokens = torch.randn(2, 7, 24)
    state = torch.randn(2, 7, 3) * 0.1
    moments = torch.randn(2, 9, 7)
    membership = torch.zeros(2, 7, 9, dtype=torch.bool)
    membership[:, :2, 0] = True
    membership[:, 2:5, 1] = True
    membership[:, 5:, 5] = True
    mask = membership.any(dim=1)
    with torch.no_grad():
        coupler.gate.fill_(1.0)
        torch.nn.init.normal_(coupler.residual[-1].weight, std=0.02)
    changed = moments.clone()
    changed[:, 0, :] += 1.0
    original = coupler(tokens, state, moments, membership, mask, torch.ones(2))
    modified = coupler(tokens, state, changed, membership, mask, torch.ones(2))
    assert not torch.equal(original[:, :2], modified[:, :2])
    assert torch.equal(original[:, 2:], modified[:, 2:])


def test_group_moment_coupler_commits_late_in_the_flow() -> None:
    coupler = GroupMomentCoupler(24, width=8)
    early = coupler.commit_weight(torch.tensor([0.0, 0.25]))
    late = coupler.commit_weight(torch.tensor([0.75, 1.0]))
    assert early.shape == late.shape == (2,)
    assert torch.all(late > early)
    assert torch.all((early >= 0.0) & (late <= 1.0))


def test_group_moment_model_preserves_identity_and_conditions_gradient() -> None:
    torch.manual_seed(43)
    base = _model(group_moments=False).eval()
    candidate = _model(group_moments=True).eval()
    candidate.load_state_dict(base.state_dict(), strict=False)
    batch = _batch()
    base_context = base.prepare_context(batch)
    candidate_context = candidate.prepare_context(batch)
    target = build_hierarchy_targets(batch, candidate_context)
    time = torch.full((2,), 0.5)
    with torch.no_grad():
        base_output = base(target, time, base_context).block_state
        candidate_output = candidate(
            target,
            time,
            candidate_context,
        ).block_state
    assert torch.equal(base_output, candidate_output)

    candidate = _model(group_moments=True).eval()
    coupler = candidate.group_moment_coupler
    relation_head = candidate.relation_head
    assert coupler is not None
    assert relation_head is not None
    with torch.no_grad():
        coupler.gate.fill_(1.0)
        torch.nn.init.normal_(coupler.residual[-1].weight, std=0.02)
        torch.nn.init.normal_(relation_head.feature_head[-1].weight, std=0.02)
        candidate.typed.cross_attention.gate.fill_(1.0)
        torch.nn.init.normal_(candidate.hierarchy_to_latent.weight, std=0.02)
        torch.nn.init.normal_(candidate.backbone.final.linear.weight, std=0.02)
    candidate_context = candidate.prepare_context(batch)
    candidate(
        target,
        time,
        candidate_context,
    ).block_state.square().mean().backward()
    assert coupler.residual[-1].weight.grad is not None
    assert coupler.residual[-1].weight.grad.abs().sum() > 0.0


def test_group_moment_checkpoint_rebuilds_opt_in_branch() -> None:
    model = _model(group_moments=True)
    checkpoint = {
        "config": {
            "architecture": "typed_hierarchy",
            "hidden": 24,
            "heads": 3,
            "depth": 1,
            "head_dim": 8,
            "typed_layers": 1,
            "structural_relations": True,
            "group_moment_conditioning": True,
        },
        "model": model.state_dict(),
    }
    restored = load_chip_checkpoint(checkpoint, "cpu", use_ema=False)
    assert isinstance(restored, HierarchicalTokenFlowModel)
    assert restored.group_moment_coupler is not None
    assert "group_moment_conditioning" in RESUME_CRITICAL_KEYS
    assert "group_moment_coupler." in RESUME_NEW_PARAM_PREFIXES


def test_native_group_moment_target_and_rollout_state() -> None:
    model = _model(group_moments=True)
    context = model.prepare_context(_batch())
    target = build_hierarchy_targets(_batch(), context)
    assert target.structural_moment_state is not None
    assert target.structural_moment_state.shape == (2, 9, 7)

    other = HierarchyState(
        global_state=torch.zeros_like(target.global_state),
        structural_state=torch.zeros_like(target.structural_state),
        block_state=torch.zeros_like(target.block_state),
        port_state=torch.zeros_like(target.port_state),
        structural_moment_state=torch.zeros_like(target.structural_moment_state),
    )
    midpoint = interpolate_hierarchy(other, target, torch.full((2,), 0.5))
    assert midpoint.structural_moment_state is not None
    assert torch.allclose(
        midpoint.structural_moment_state,
        target.structural_moment_state * 0.5,
    )

    prediction = HierarchyPrediction(
        global_state=target.global_state,
        structural_state=target.structural_state,
        block_state=target.block_state,
        port_state=target.port_state,
        structural_moment_state=target.structural_moment_state,
    )
    advanced = advance_hierarchy_state(
        other,
        prediction,
        torch.ones(2),
        step_size=1.0,
        port_mask=context.typed.port_mask,
    )
    assert advanced.structural_moment_state is not None
    assert torch.allclose(
        advanced.structural_moment_state,
        target.structural_moment_state,
    )


def test_native_group_moment_prediction_stays_masked_to_active_slots() -> None:
    model = _model(group_moments=True)
    context = model.prepare_context(_batch())
    target = build_hierarchy_targets(_batch(), context)
    module = model.group_moment_coupler
    assert module is not None
    with torch.no_grad():
        module.native_residual[-1].weight.normal_(std=0.1)
        module.native_residual[-1].bias.fill_(0.5)

    moments, _residual = apply_native_group_moment(
        module,
        context.typed.block_tokens,
        target.block_state,
        context.typed.structural_tokens,
        target.structural_state,
        target.structural_moment_state,
        context.coarse_membership,
        context.typed.structural_mask,
        torch.full((2,), 0.5),
    )

    inactive = ~context.typed.structural_mask
    assert inactive.any()
    assert torch.count_nonzero(moments[inactive]) == 0


def test_group_moment_loss_ignores_inactive_slots() -> None:
    model = _model(group_moments=True)
    context = model.prepare_context(_batch())
    target = build_hierarchy_targets(_batch(), context)
    assert target.structural_moment_state is not None
    inactive = ~context.typed.structural_mask
    prediction_moments = target.structural_moment_state.clone()
    prediction_moments[inactive] = 100.0
    prediction = HierarchyPrediction(
        global_state=target.global_state,
        structural_state=target.structural_state,
        block_state=target.block_state,
        port_state=target.port_state,
        structural_moment_state=prediction_moments,
    )
    _loss, terms = hierarchy_flow_loss(
        prediction,
        replace(target),
        huber_beta=0.1,
        global_weight=1.0,
        structural_weight=1.0,
        port_weight=1.0,
        structural_mask=context.typed.structural_mask,
    )

    assert terms["flow_structural_moment"] == 0
