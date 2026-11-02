from __future__ import annotations

import torch

from floorflow.chip_flow.backbone_lora import LoRALinear
from floorflow.chip_flow.hierarchical_flow import (
    HierarchicalTokenFlowModel,
    build_hierarchy_targets,
)


def _batch() -> dict:
    count = 5
    constraints = torch.zeros(2, count, 5)
    constraints[:, 0, 4] = 1
    constraints[:, 1, 4] = 2
    return {
        "areas": torch.ones(2, count),
        "boxes": torch.tensor(
            [[[index, 0.0, 1.0, 1.0] for index in range(count)]] * 2
        ),
        "constraints": constraints,
        "features": torch.zeros(2, count, 10),
        "target_latent": torch.randn(2, count, 3),
        "source_latent": torch.randn(2, count, 3),
        "b2b": torch.tensor([[[0.0, 1.0, 1.0]]] * 2),
        "p2b": torch.tensor([[[0.0, 0.0, 1.0]]] * 2),
        "pins": torch.randn(2, 2, 2),
        "n_b2b": torch.tensor([1, 1]),
        "n_p2b": torch.tensor([1, 1]),
        "n_pins": torch.tensor([1, 1]),
    }


def test_lora_linear_starts_exact_and_only_delta_is_trainable() -> None:
    base = torch.nn.Linear(6, 4)
    values = torch.randn(3, 6)
    adapted = LoRALinear(base, rank=2, alpha=2.0)
    assert torch.equal(adapted(values), base(values))
    assert not adapted.weight.requires_grad
    assert adapted.lora_a.requires_grad and adapted.lora_b.requires_grad
    adapted.lora_b.data.normal_(std=0.02)
    assert not torch.equal(adapted(values), base(values))


def test_model_lora_preserves_m27_output_at_initialization() -> None:
    base = HierarchicalTokenFlowModel(
        hidden=64,
        heads=4,
        depth=2,
        typed_layers=1,
        head_dim=64,
    )
    adapted = HierarchicalTokenFlowModel(
        hidden=64,
        heads=4,
        depth=2,
        typed_layers=1,
        head_dim=64,
        backbone_lora_rank=2,
    )
    incompatible = adapted.load_state_dict(base.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert incompatible.missing_keys
    assert all(
        name.endswith(("lora_a", "lora_b"))
        for name in incompatible.missing_keys
    )
    batch = _batch()
    context = adapted.prepare_context(batch)
    target = build_hierarchy_targets(batch, context)
    time = torch.full((2,), 0.4)
    assert torch.equal(
        base(target, time, context).block_state,
        adapted(target, time, context).block_state,
    )
    assert any(
        name.startswith("backbone.blocks.")
        and name.endswith("lora_a")
        and parameter.requires_grad
        for name, parameter in adapted.named_parameters()
    )
    assert not adapted.backbone.blocks[0].attn.qkv.weight.requires_grad
