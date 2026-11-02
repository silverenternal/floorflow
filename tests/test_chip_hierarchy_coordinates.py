from __future__ import annotations

import torch

from floorflow.chip_flow.checkpoint_loading import load_chip_checkpoint
from floorflow.chip_flow.hierarchical_flow import HierarchicalTokenFlowModel
from floorflow.chip_flow.hierarchy_coordinates import (
    build_hierarchy_coordinate_plan,
    parameterize_hierarchy_block_input,
    restore_hierarchy_coordinate_prediction,
    state_from_hierarchy_coordinates,
    state_to_hierarchy_coordinates,
)
from floorflow.chip_flow.hierarchy_targets import build_hierarchy_targets
from floorflow.chip_flow.training_runtime import _is_no_decay


def _coordinate_inputs(
    shift: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    structure = torch.zeros(1, 9, 3)
    structure[0, 0, :2] = torch.tensor([2.0 + shift, 3.0])
    structure[0, 1, :2] = torch.tensor([8.0 + shift, 3.0])
    membership = torch.zeros(1, 4, 9, dtype=torch.bool)
    membership[0, :2, 0] = True
    membership[0, 2:, 1] = True
    mask = torch.zeros(1, 9, dtype=torch.bool)
    mask[0, :2] = True
    blocks = torch.tensor(
        [
            [
                [1.5 + shift, 3.0, 0.0],
                [2.5 + shift, 3.0, 0.1],
                [7.5 + shift, 3.0, -0.1],
                [8.5 + shift, 3.0, 0.0],
            ]
        ]
    )
    global_state = torch.tensor([[5.0 + shift, 3.0, 12.0, 6.0]])
    return blocks, global_state, structure, membership, mask


def test_hierarchy_coordinate_round_trip_and_cluster_selection() -> None:
    blocks, global_state, structure, membership, mask = _coordinate_inputs()
    plan = build_hierarchy_coordinate_plan(
        blocks,
        global_state,
        structure,
        membership,
        mask,
    )
    assert plan.cluster_slot.tolist() == [[0, 0, 1, 1]]
    local = state_to_hierarchy_coordinates(blocks, plan)
    restored = state_from_hierarchy_coordinates(local, plan)
    assert torch.allclose(restored, blocks, atol=1.0e-6)
    assert torch.equal(local[..., 2], blocks[..., 2])


def test_cluster_local_coordinates_are_translation_invariant() -> None:
    original = _coordinate_inputs()
    shifted = _coordinate_inputs(shift=4.0)
    plans = [
        build_hierarchy_coordinate_plan(blocks, global_state, structure, membership, mask)
        for blocks, global_state, structure, membership, mask in (original, shifted)
    ]
    local = [
        state_to_hierarchy_coordinates(blocks, plan)
        for (blocks, *_), plan in zip((original, shifted), plans)
    ]
    assert torch.allclose(local[0], local[1], atol=1.0e-6)


def test_envelope_only_mode_ignores_cluster_frames() -> None:
    blocks, global_state, structure, membership, mask = _coordinate_inputs()
    plan = build_hierarchy_coordinate_plan(
        blocks,
        global_state,
        structure,
        membership,
        mask,
        cluster_local=False,
    )
    assert torch.equal(plan.cluster_slot, torch.full_like(plan.cluster_slot, -1))
    local = state_to_hierarchy_coordinates(blocks, plan)
    expected = (blocks[..., :2] - global_state[:, None, :2]) / global_state[
        :, None, 2:4
    ]
    assert torch.allclose(local[..., :2], expected, atol=1.0e-6)


def test_coordinate_gate_starts_at_identity_and_reaches_local_frame() -> None:
    blocks, global_state, structure, membership, mask = _coordinate_inputs()
    local_zero, plan_zero = parameterize_hierarchy_block_input(
        blocks,
        global_state,
        structure,
        membership,
        mask,
        enabled=True,
        blend=0.0,
    )
    local_one, plan_one = parameterize_hierarchy_block_input(
        blocks,
        global_state,
        structure,
        membership,
        mask,
        enabled=True,
        blend=1.0,
    )
    assert torch.equal(local_zero, blocks)
    assert plan_zero is not None
    assert plan_one is not None
    assert torch.allclose(
        local_one,
        state_to_hierarchy_coordinates(blocks, plan_one),
    )
    assert torch.equal(
        restore_hierarchy_coordinate_prediction(blocks, plan_zero, blend=0.0),
        blocks,
    )


def test_coordinate_parameterization_runs_in_hierarchy_forward_and_preserves_state_grad() -> None:
    torch.manual_seed(31)
    model = HierarchicalTokenFlowModel(
        hidden=32,
        heads=4,
        depth=1,
        head_dim=16,
        typed_layers=1,
        hierarchy_coordinate_parameterization=True,
    )
    model.coordinate_gate.bias.data.fill_(1.0)
    count = 5
    constraints = torch.zeros(1, count, 5)
    constraints[0, :2, 3] = 1
    constraints[0, 2:4, 3] = 2
    boxes = torch.tensor(
        [[[1.0, 3.0, 1.0, 0.5], [3.0, 3.0, 1.0, 0.5],
          [7.0, 3.0, 1.0, 0.5], [9.0, 3.0, 1.0, 0.5],
          [5.0, 6.0, 1.0, 0.5]]]
    )
    batch = {
        "areas": torch.full((1, count), 0.5),
        "boxes": boxes,
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
    context = model.prepare_context(batch)
    target = build_hierarchy_targets(batch, context)
    target = type(target)(
        global_state=target.global_state.requires_grad_(True),
        structural_state=target.structural_state.requires_grad_(True),
        block_state=target.block_state,
        port_state=target.port_state,
    )
    prediction = model(target, torch.full((1,), 0.5), context)
    assert torch.isfinite(prediction.block_state).all()
    prediction.block_state.square().mean().backward()
    assert target.global_state.grad is not None
    assert target.structural_state.grad is not None
    assert target.global_state.grad.abs().sum() > 0
    assert target.structural_state.grad.abs().sum() > 0


def test_coordinate_branch_loads_old_state_at_identity() -> None:
    base = HierarchicalTokenFlowModel(
        hidden=16,
        heads=4,
        depth=1,
        head_dim=16,
        typed_layers=1,
    )
    checkpoint = {
        "config": {
            "architecture": "typed_hierarchy",
            "hidden": 16,
            "heads": 4,
            "depth": 1,
            "head_dim": 16,
            "typed_layers": 1,
            "hierarchy_coordinate_parameterization": True,
        },
        "model": base.state_dict(),
        "ema": base.state_dict(),
    }
    restored = load_chip_checkpoint(checkpoint, "cpu", use_ema=True)
    assert isinstance(restored, HierarchicalTokenFlowModel)
    assert restored.coordinate_gate is not None
    assert restored.coordinate_gate.bias.item() == 0.0
    assert _is_no_decay("coordinate_gate.scorer.1.weight")
