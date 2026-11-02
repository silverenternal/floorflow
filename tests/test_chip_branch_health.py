from __future__ import annotations

import torch
from torch import nn

from floorflow.chip_flow.branch_health import branch_health, dead_branches
from floorflow.chip_flow.typed_graph_encoder import TypedGraphEncoder


def test_branch_health_detects_dead_and_active_zero_gates() -> None:
    model = TypedGraphEncoder(hidden=16, heads=4, graph_layers=1)
    batch = {
        "constraints": torch.zeros(1, 3, 5),
        "areas": torch.ones(1, 3),
        "b2b": torch.tensor([[[0.0, 1.0, 1.0]]]),
        "p2b": torch.tensor([[[0.0, 0.0, 1.0]]]),
        "pins": torch.tensor([[[0.2, 0.5]]]),
        "n_b2b": torch.ones(1, dtype=torch.long),
        "n_p2b": torch.ones(1, dtype=torch.long),
        "n_pins": torch.ones(1, dtype=torch.long),
    }
    features = torch.zeros(1, 3, 19)
    output = model(features, batch["constraints"], **{
        key: value for key, value in batch.items() if key != "constraints"
    })
    output.block_tokens.square().mean().backward()
    initial = branch_health(model)
    assert "local_geometry" in initial
    assert "typed_cross_attention" in initial
    assert "local_geometry" in dead_branches(initial)

    model.local_geometry_encoder.gate.data.fill_(1.0)
    model.cross_attention.gate.data.fill_(1.0)
    model.zero_grad(set_to_none=True)
    output = model(features, batch["constraints"], **{
        key: value for key, value in batch.items() if key != "constraints"
    })
    output.block_tokens.square().mean().backward()
    active = branch_health(model)
    assert active["local_geometry"]["active"]
    assert active["typed_cross_attention"]["active"]


def test_branch_health_uses_gradient_reachability_for_ungated_branches() -> None:
    model = nn.Module()
    model.structured_heads = nn.Linear(2, 2)
    loss = model.structured_heads(torch.ones(1, 2)).square().mean()
    loss.backward()
    health = branch_health(model)
    assert health["structured_heads"]["gate_present"] is False
    assert health["structured_heads"]["active"] is True
