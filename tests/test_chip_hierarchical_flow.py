from __future__ import annotations

import io

import pytest
import torch

from floorflow.chip_flow.checkpoint_loading import load_chip_checkpoint
from floorflow.chip_flow.flow_adapter import sample_for_batch
from floorflow.chip_flow.hierarchical_flow import (
    HierarchicalTokenFlowModel,
    build_hierarchy_targets,
    outline_condition_from_boxes,
    outline_condition_from_hw,
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


def _model() -> HierarchicalTokenFlowModel:
    return HierarchicalTokenFlowModel(hidden=192, heads=6)


def test_hierarchy_targets_and_zero_init_block_equivalence() -> None:
    torch.manual_seed(3)
    model = _model()
    batch = _batch()
    context = model.prepare_context(batch)
    target = build_hierarchy_targets(batch, context)
    assert target.global_state.shape == (2, 4)
    assert target.structural_state.shape == (2, 9, 3)
    assert target.block_state.shape == (2, 7, 3)
    assert target.port_state.shape == (2, 3, 2)
    time = torch.full((2,), 0.5)
    prediction = model(target, time, context)
    base = model.backbone(
        target.block_state,
        time,
        context.denoiser_cond,
    )
    assert torch.equal(prediction.block_state, base)
    assert prediction.global_state.shape == (2, 4)
    assert prediction.structural_state.shape == (2, 9, 3)
    assert prediction.port_state.shape == (2, 3, 2)


def test_hierarchy_context_does_not_use_target_latent() -> None:
    model = _model().eval()
    batch = _batch()
    original = model.prepare_context(batch)
    changed = {key: value.clone() for key, value in batch.items()}
    changed["target_latent"].add_(10.0)
    rerun = model.prepare_context(changed)
    assert torch.equal(
        original.typed.block_tokens,
        rerun.typed.block_tokens,
    )
    assert torch.equal(
        original.denoiser_cond.features,
        rerun.denoiser_cond.features,
    )


def test_hierarchy_residual_couples_states_and_receives_gradient() -> None:
    torch.manual_seed(7)
    model = _model()
    model.typed.cross_attention.gate.data.fill_(1.0)
    torch.nn.init.normal_(model.hierarchy_to_latent.weight, std=0.02)
    torch.nn.init.normal_(model.backbone.final.linear.weight, std=0.02)
    batch = _batch()
    context = model.prepare_context(batch)
    target = build_hierarchy_targets(batch, context)
    prediction = model(target, torch.full((2,), 0.4), context)
    base = model.backbone(
        target.block_state,
        torch.full((2,), 0.4),
        context.denoiser_cond,
    )
    assert not torch.equal(prediction.block_state, base)
    (
        prediction.block_state.square().mean()
        + prediction.global_state.square().mean()
    ).backward()
    assert model.typed.port_input.weight.grad is not None
    assert model.global_head[-1].weight.grad is not None
    assert model.hierarchy_to_latent.weight.grad is not None


def test_hierarchy_block_flow_is_permutation_equivariant() -> None:
    torch.manual_seed(11)
    model = _model().eval()
    torch.nn.init.normal_(model.hierarchy_to_latent.weight, std=0.01)
    model.typed.cross_attention.gate.data.fill_(1.0)
    batch = _batch()
    context = model.prepare_context(batch)
    target = build_hierarchy_targets(batch, context)
    original = model(target, torch.ones(2), context).block_state
    permutation = torch.tensor([3, 0, 5, 1, 6, 2, 4])
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(7)
    permuted = {key: value.clone() for key, value in batch.items()}
    permuted["b2b"] = torch.cat(
        [
            inverse[batch["b2b"][..., :2].long()].float(),
            batch["b2b"][..., 2:],
        ],
        dim=-1,
    )
    permuted["p2b"] = batch["p2b"].clone()
    permuted["p2b"][..., 1] = inverse[batch["p2b"][..., 1].long()].float()
    for key in ("areas", "boxes", "constraints", "features", "target_latent"):
        permuted[key] = batch[key][:, permutation]
    context = model.prepare_context(permuted)
    target = build_hierarchy_targets(permuted, context)
    reordered = model(target, torch.ones(2), context).block_state
    assert torch.allclose(
        original[:, permutation],
        reordered,
        atol=2.0e-4,
    )


def test_hierarchy_checkpoint_round_trip() -> None:
    model = _model()
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = _model()
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    for key, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key]), key


def test_dynamic_graph_checkpoint_rebuilds_opt_in_branch() -> None:
    model = HierarchicalTokenFlowModel(
        hidden=64,
        heads=4,
        depth=2,
        typed_layers=1,
        head_dim=64,
        dynamic_graph_relay=True,
    )
    checkpoint = {
        "config": {
            "architecture": "typed_hierarchy",
            "hidden": 64,
            "heads": 4,
            "depth": 2,
            "head_dim": 64,
            "typed_layers": 1,
            "dynamic_graph_relay": True,
        },
        "ema": model.state_dict(),
        "model": model.state_dict(),
    }
    restored = load_chip_checkpoint(checkpoint, "cpu", use_ema=True)
    assert isinstance(restored, HierarchicalTokenFlowModel)
    assert restored.dynamic_relay is not None


def test_hierarchy_parameter_budget() -> None:
    model = _model()
    assert sum(parameter.numel() for parameter in model.parameters()) < 9_000_000


def test_dynamic_graph_relay_is_zero_initialized_and_receives_gradient() -> None:
    torch.manual_seed(17)
    base = _model().eval()
    torch.nn.init.normal_(base.backbone.final.linear.weight, std=0.02)
    dynamic = HierarchicalTokenFlowModel(
        hidden=192,
        heads=6,
        dynamic_graph_relay=True,
    ).eval()
    dynamic.load_state_dict(base.state_dict(), strict=False)
    batch = _batch()
    base_context = base.prepare_context(batch)
    dynamic_context = dynamic.prepare_context(batch)
    targets = build_hierarchy_targets(batch, base_context)
    assert torch.equal(
        base(targets, torch.full((2,), 0.5), base_context).block_state,
        dynamic(targets, torch.full((2,), 0.5), dynamic_context).block_state,
    )
    assert dynamic.dynamic_relay is not None
    assert dynamic.dynamic_relay.gate.item() == 0
    dynamic.dynamic_relay.gate.data.fill_(1.0)
    dynamic(
        targets,
        torch.full((2,), 0.5),
        dynamic_context,
    ).block_state.square().mean().backward()
    assert dynamic.dynamic_relay.output.weight.grad is not None
    assert dynamic.dynamic_relay.output.weight.grad.abs().sum() > 0


def test_dynamic_graph_relay_depends_on_current_trajectory_state() -> None:
    torch.manual_seed(19)
    model = HierarchicalTokenFlowModel(
        hidden=64,
        heads=4,
        dynamic_graph_relay=True,
    ).eval()
    batch = _batch()
    context = model.prepare_context(batch)
    state = build_hierarchy_targets(batch, context).block_state
    assert context.edge_state is not None
    assert model.dynamic_relay is not None
    model.dynamic_relay.gate.data.fill_(1.0)
    torch.nn.init.normal_(model.dynamic_relay.output.weight, std=0.02)
    first = model.dynamic_relay(
        state,
        cache=context.edge_state.cache,
    )
    second = model.dynamic_relay(
        state + 0.25,
        cache=context.edge_state.cache,
    )
    assert not torch.equal(first, second)


def test_dynamic_graph_input_opens_from_parent_without_deadlock() -> None:
    torch.manual_seed(29)
    base = HierarchicalTokenFlowModel(
        hidden=24,
        heads=3,
        depth=1,
        head_dim=8,
        typed_layers=1,
    ).eval()
    torch.nn.init.normal_(base.backbone.final.linear.weight, std=0.02)
    torch.manual_seed(29)
    dynamic = HierarchicalTokenFlowModel(
        hidden=24,
        heads=3,
        depth=1,
        head_dim=8,
        typed_layers=1,
        dynamic_graph_input=True,
    ).eval()
    dynamic.load_state_dict(base.state_dict(), strict=False)
    batch = _batch()
    base_context = base.prepare_context(batch)
    dynamic_context = dynamic.prepare_context(batch)
    target = build_hierarchy_targets(batch, base_context)
    time = torch.full((2,), 0.5)
    with torch.no_grad():
        torch.testing.assert_close(
            base(target, time, base_context).block_state,
            dynamic(target, time, dynamic_context).block_state,
        )

    prediction = dynamic(
        target,
        time,
        dynamic_context,
    )
    (prediction.block_state - target.block_state).square().mean().backward()

    assert dynamic.dynamic_relay.gate.grad is not None
    assert dynamic.dynamic_relay.gate.grad.abs().sum() > 0


def test_structural_relations_modulate_l1_prediction_before_supervision() -> None:
    torch.manual_seed(31)
    base = HierarchicalTokenFlowModel(
        hidden=24,
        heads=3,
        depth=1,
        head_dim=8,
        typed_layers=1,
    ).eval()
    torch.manual_seed(31)
    relational = HierarchicalTokenFlowModel(
        hidden=24,
        heads=3,
        depth=1,
        head_dim=8,
        typed_layers=1,
        structural_relations=True,
    ).eval()
    relational.load_state_dict(base.state_dict(), strict=False)
    relational.relation_head.gate.data.fill_(1.0)
    batch = _batch()
    base_context = base.prepare_context(batch)
    relational_context = relational.prepare_context(batch)
    target = build_hierarchy_targets(batch, base_context)
    time = torch.full((2,), 0.5)
    with torch.no_grad():
        base_prediction = base(target, time, base_context)
        relational_prediction = relational(target, time, relational_context)
    assert not torch.allclose(
        base_prediction.structural_state,
        relational_prediction.structural_state,
    )

    relational_prediction = relational(target, time, relational_context)
    (
        relational_prediction.structural_state
        - target.structural_state
    ).square().mean().backward()

    assert relational.relation_head.gate.grad is not None
    assert relational.relation_head.gate.grad.abs().sum() > 0


def test_outline_condition_helpers_normalize_by_block_area() -> None:
    boxes = torch.tensor(
        [
            [
                [0.0, 0.0, 2.0, 1.0],
                [2.0, 1.0, 2.0, 1.0],
            ]
        ]
    )
    areas = torch.tensor([[2.0, 2.0]])
    scale = areas.sum(dim=1).sqrt()

    from_boxes = outline_condition_from_boxes(boxes, areas, margin=0.1)
    from_hw = outline_condition_from_hw(
        torch.tensor([[4.8, 2.4]]),
        areas,
    )

    torch.testing.assert_close(from_boxes[0, 2:], from_hw[0, 2:])
    torch.testing.assert_close(
        from_hw,
        torch.tensor([[1.2, 0.6, 2.4, 1.2]]),
    )
    assert from_boxes[0, 2] == torch.tensor(4.8) / scale[0]


def test_outline_conditioning_disabled_path_has_no_parameters() -> None:
    model = _model()
    batch = _batch()
    context = model.prepare_context(batch)

    assert context.outline_state is None
    assert model.outline_to_hidden is None


def test_enabled_outline_path_is_zero_initialized_equivalence() -> None:
    batch = _batch()
    batch["outline_state"] = outline_condition_from_boxes(
        batch["boxes"],
        batch["areas"],
        margin=0.1,
    )
    torch.manual_seed(5)
    base = HierarchicalTokenFlowModel(hidden=192, heads=6)
    torch.manual_seed(5)
    conditioned = HierarchicalTokenFlowModel(
        hidden=192,
        heads=6,
        outline_conditioning=True,
    )
    base_context = base.prepare_context(batch)
    conditioned_context = conditioned.prepare_context(batch)
    target = build_hierarchy_targets(batch, base_context)
    time = torch.full((2,), 0.5)

    base_prediction = base(target, time, base_context)
    conditioned_prediction = conditioned(target, time, conditioned_context)

    torch.testing.assert_close(
        conditioned_prediction.block_state,
        base_prediction.block_state,
    )
    torch.testing.assert_close(
        conditioned_prediction.global_state,
        base_prediction.global_state,
    )


def test_enabled_outline_conditioning_requires_state() -> None:
    model = HierarchicalTokenFlowModel(
        hidden=192,
        heads=6,
        outline_conditioning=True,
    )

    try:
        model.prepare_context(_batch())
    except ValueError as error:
        assert "outline_state" in str(error)
    else:
        raise AssertionError("missing outline_state must be rejected")


def test_enabled_outline_projection_receives_gradient() -> None:
    torch.manual_seed(13)
    model = HierarchicalTokenFlowModel(
        hidden=192,
        heads=6,
        outline_conditioning=True,
    )
    batch = _batch()
    batch["outline_state"] = outline_condition_from_boxes(
        batch["boxes"],
        batch["areas"],
        margin=0.1,
    )
    context = model.prepare_context(batch)
    target = build_hierarchy_targets(batch, context)
    prediction = model(target, torch.full((2,), 0.5), context)

    (prediction.block_state.square().mean() + prediction.global_state.square().mean()).backward()

    gradient = model.outline_to_hidden.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0.0


def test_outline_conditioning_checkpoint_round_trip() -> None:
    torch.manual_seed(17)
    model = HierarchicalTokenFlowModel(
        hidden=192,
        heads=6,
        outline_conditioning=True,
    )
    checkpoint = {
        "model": model.state_dict(),
        "ema": model.state_dict(),
        "config": {
            "architecture": "typed_hierarchy",
            "hidden": 192,
            "heads": 6,
            "depth": 12,
            "head_dim": 64,
            "graph_pe_k": 2,
            "typed_layers": 2,
            "outline_conditioning": True,
        },
    }

    restored = load_chip_checkpoint(checkpoint, "cpu", use_ema=False)

    assert restored.outline_conditioning
    assert restored.outline_to_hidden is not None
    assert restored.outline_to_hidden.weight.shape == (192, 4)


def test_outline_conditioning_adds_only_the_projection_parameters() -> None:
    base = HierarchicalTokenFlowModel(hidden=192, heads=6)
    conditioned = HierarchicalTokenFlowModel(
        hidden=192,
        heads=6,
        outline_conditioning=True,
    )

    base_parameters = sum(parameter.numel() for parameter in base.parameters())
    conditioned_parameters = sum(
        parameter.numel() for parameter in conditioned.parameters()
    )

    assert conditioned_parameters - base_parameters == 4 * 192 + 192


def test_outline_conditioned_sampling_rejects_ground_truth_fallback() -> None:
    torch.manual_seed(31)
    model = HierarchicalTokenFlowModel(
        hidden=192,
        heads=6,
        outline_conditioning=True,
    )
    batch = _batch()
    batch["source_latent"] = torch.randn(2, 7, 3)
    assert "outline_state" not in batch

    with pytest.raises(ValueError, match="target-free"):
        sample_for_batch(
            model,
            batch,
            sampling_steps=1,
            generator=torch.Generator().manual_seed(31),
        )


def test_outline_projection_initialization_preserves_rng_sequence() -> None:
    torch.manual_seed(29)
    HierarchicalTokenFlowModel(hidden=192, heads=6)
    base_random = torch.rand(8)

    torch.manual_seed(29)
    HierarchicalTokenFlowModel(
        hidden=192,
        heads=6,
        outline_conditioning=True,
    )
    condition_random = torch.rand(8)

    torch.testing.assert_close(base_random, condition_random, rtol=0, atol=0)


