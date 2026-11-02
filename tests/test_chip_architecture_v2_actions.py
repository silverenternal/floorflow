from __future__ import annotations

import torch

from floorflow.chip_flow.checkpoint_loading import load_chip_checkpoint
from floorflow.chip_flow.hierarchical_flow import (
    HierarchicalTokenFlowModel,
    build_hierarchy_targets,
)


def _batch() -> dict:
    count = 6
    boxes = torch.zeros(2, count, 4)
    boxes[..., 2:] = 1.0
    boxes[..., 0] = torch.arange(count) * 1.2
    constraints = torch.zeros(2, count, 5)
    constraints[..., 3] = torch.tensor([1, 1, 2, 2, 3, 3])
    constraints[:, :2, 4] = 1
    return {
        "areas": boxes[..., 2] * boxes[..., 3],
        "boxes": boxes,
        "constraints": constraints,
        "features": torch.zeros(2, count, 19),
        "target_latent": torch.randn(2, count, 3),
        "b2b": torch.tensor(
            [[[0.0, 1.0, 1.0], [1.0, 2.0, 0.5]]] * 2
        ),
        "p2b": torch.tensor(
            [[[0.0, 0.0, 1.0], [1.0, 1.0, 0.5]]] * 2
        ),
        "pins": torch.rand(2, 2, 2),
        "n_b2b": torch.tensor([2, 2]),
        "n_p2b": torch.tensor([2, 2]),
        "n_pins": torch.tensor([2, 2]),
    }


def test_legalizer_response_can_act_on_the_endpoint() -> None:
    torch.manual_seed(13)
    batch = _batch()
    base = HierarchicalTokenFlowModel(
        hidden=32,
        heads=4,
        depth=1,
        head_dim=16,
        typed_layers=1,
        structured_heads=True,
        projector_grid_size=8,
    )
    acting = HierarchicalTokenFlowModel(
        hidden=32,
        heads=4,
        depth=1,
        head_dim=16,
        typed_layers=1,
        structured_heads=True,
        structured_legalizer_apply=True,
        structured_legalizer_strength=0.5,
        projector_grid_size=8,
    )
    acting.load_state_dict(base.state_dict(), strict=False)
    assert acting.structured_heads is not None
    torch.nn.init.normal_(
        acting.structured_heads.legalizer_response.displacement.weight,
        std=0.05,
    )
    acting.structured_heads.legalizer_response.displacement.bias.data.fill_(
        100.0
    )
    context = acting.prepare_context(batch)
    target = build_hierarchy_targets(batch, context)
    unchanged = base(
        target,
        torch.full((2,), 0.5),
        base.prepare_context(batch),
    )
    action_disabled = acting(
        target,
        torch.full((2,), 0.5),
        context,
        apply_legalizer_action=False,
    )
    acted = acting(
        target,
        torch.full((2,), 0.5),
        context,
        apply_legalizer_action=True,
    )
    torch.testing.assert_close(
        unchanged.block_state,
        action_disabled.block_state,
    )
    assert not torch.equal(unchanged.block_state, acted.block_state)
    displacement = (
        acted.block_state - action_disabled.block_state
    ).norm(dim=-1)
    assert displacement.max() <= 0.05 + 1.0e-6


def test_v2_checkpoint_loader_accepts_zero_gated_parent_conversion() -> None:
    base = HierarchicalTokenFlowModel(
        hidden=16,
        heads=4,
        depth=1,
        head_dim=8,
        typed_layers=1,
        projector_grid_size=8,
    )
    checkpoint = {
        "config": {
            "architecture": "typed_hierarchy",
            "hidden": 16,
            "heads": 4,
            "depth": 1,
            "head_dim": 8,
            "typed_layers": 1,
            "relation_specific": True,
            "structural_relations": True,
            "complexity_conditioning": True,
            "degree_expert_conditioning": True,
            "dynamic_graph_input": True,
            "spatial_route_field": True,
            "structured_heads": True,
            "structured_legalizer_apply": True,
            "projector_grid_size": 8,
        },
        "model": base.state_dict(),
        "ema": base.state_dict(),
    }
    restored = load_chip_checkpoint(checkpoint, "cpu", use_ema=True)
    assert restored.relation_head is not None
    assert restored.structured_legalizer_apply
