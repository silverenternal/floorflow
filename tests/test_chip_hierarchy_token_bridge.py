from __future__ import annotations

from dataclasses import replace

import torch
from safetensors.torch import save_file
from trinity.models.backbone import DenoiserCond

from floorflow.chip_flow.checkpoint_loading import load_chip_checkpoint
from floorflow.chip_flow.hierarchical_flow import HierarchicalTokenFlowModel
from floorflow.chip_flow.hierarchy_token_bridge import (
    HierarchyTokenBridge,
    HierarchyTokenBridgeInput,
    TokenBridgeBackbone,
)
from floorflow.chip_flow.trinity_backbone_init import initialize_trinity_backbone


def _bridge_input(batch: int = 2, count: int = 5, hidden: int = 48):
    slots = 9
    return HierarchyTokenBridgeInput(
        block_tokens=torch.randn(batch, count, hidden),
        block_state=torch.randn(batch, count, 3),
        global_prediction=torch.randn(batch, 4),
        structural_prediction=torch.randn(batch, slots, 3),
        structural_tokens=torch.randn(batch, slots, hidden),
        structural_mask=torch.tensor(
            [[True, True, False, False, False, True, False, False, False]]
            * batch
        ),
        time=torch.full((batch,), 0.4),
        membership=torch.zeros(batch, count, 9, dtype=torch.bool),
    )


def test_hierarchy_token_bridge_is_zero_initialized_and_trainable() -> None:
    torch.manual_seed(3)
    bridge = HierarchyTokenBridge(48, 3)
    inputs = _bridge_input()
    output = bridge(inputs)
    assert torch.equal(output, torch.zeros_like(output))
    output.sum().backward()
    assert bridge.update[-1].weight.grad is not None
    assert bridge.update[-1].weight.grad.abs().sum() > 0
    bridge.zero_grad(set_to_none=True)
    bridge.update[-1].weight.data.normal_(std=0.02)
    assert not torch.equal(bridge(inputs), output)


def test_bounded_hierarchy_bridge_starts_at_zero_and_is_finite() -> None:
    bridge = HierarchyTokenBridge(48, 3, bounded=True)
    inputs = _bridge_input()
    output = bridge(inputs)
    assert torch.equal(output, torch.zeros_like(output))
    output.sum().backward()
    assert bridge.gate.grad is not None
    assert bridge.gate.grad.abs().sum() > 0
    bridge.gate.data.fill_(1.0)
    assert bridge(inputs).abs().max() <= 1.0


def test_structural_bridge_only_updates_member_blocks() -> None:
    bridge = HierarchyTokenBridge(48, 3, structural_only=True)
    membership = torch.zeros(2, 5, 9, dtype=torch.bool)
    membership[0, 0, 0] = True
    membership[1, 3, 5] = True
    inputs = replace(_bridge_input(), membership=membership)
    assert torch.equal(bridge(inputs), torch.zeros(2, 5, 48))
    bridge.update[-1].weight.data.normal_(std=0.02)
    output = bridge(inputs)
    assert output[0, 0].abs().sum() > 0
    assert output[1, 3].abs().sum() > 0
    assert output[0, 1:].abs().sum() == 0


def test_token_bridge_backbone_preserves_base_path_at_initialization() -> None:
    torch.manual_seed(5)
    model = HierarchicalTokenFlowModel(
        hidden=48,
        heads=3,
        depth=1,
        head_dim=16,
        typed_layers=1,
        hierarchy_token_bridge=True,
    )
    assert isinstance(model.backbone, TokenBridgeBackbone)
    condition = DenoiserCond(
        features=torch.randn(2, 5, 19),
        adjacency=torch.zeros(2, 5, 5),
        key_pad_mask=torch.ones(2, 5, dtype=torch.bool),
        graph_pe=torch.zeros(2, 5, 2),
    )
    torch.nn.init.normal_(model.backbone.final.linear.weight, std=0.02)
    state = torch.randn(2, 5, 3)
    time = torch.full((2,), 0.4)
    base = model.backbone(state, time, condition)
    bridged = model.backbone(
        state,
        time,
        condition,
        bridge=_bridge_input(),
    )
    assert torch.equal(base, bridged)
    model.backbone.hierarchy_bridge.update[-1].weight.data.normal_(std=0.02)
    model.backbone.hierarchy_bridge.gate.data.fill_(1.0)
    changed = model.backbone(
        state,
        time,
        condition,
        bridge=_bridge_input(),
    )
    assert not torch.equal(base, changed)


def test_hierarchy_token_bridge_checkpoint_rebuilds_branch() -> None:
    model = HierarchicalTokenFlowModel(
        hidden=48,
        heads=3,
        depth=1,
        head_dim=16,
        typed_layers=1,
        hierarchy_token_bridge=True,
    )
    checkpoint = {
        "config": {
            "architecture": "typed_hierarchy",
            "hidden": 48,
            "heads": 3,
            "depth": 1,
            "head_dim": 16,
            "typed_layers": 1,
            "hierarchy_token_bridge": True,
        },
        "ema": model.state_dict(),
        "model": model.state_dict(),
    }
    restored = load_chip_checkpoint(checkpoint, "cpu", use_ema=True)
    assert isinstance(restored, HierarchicalTokenFlowModel)
    assert isinstance(restored.backbone, TokenBridgeBackbone)


def test_trinity_initializer_accepts_only_fresh_bridge_parameters(
    tmp_path,
) -> None:
    model = HierarchicalTokenFlowModel(
        hidden=48,
        heads=3,
        depth=1,
        head_dim=16,
        typed_layers=1,
        hierarchy_token_bridge=True,
    )
    backbone_state = {
        name: value
        for name, value in model.backbone.state_dict().items()
        if not name.startswith("hierarchy_bridge.")
    }
    release = tmp_path / "release"
    release.mkdir()
    save_file(backbone_state, release / "model.safetensors")
    provenance = initialize_trinity_backbone(model, release)
    assert provenance["strict_load"] is True
    assert provenance["fresh_bridge_parameters"] > 0
