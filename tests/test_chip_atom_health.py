from __future__ import annotations

import torch
from torch import nn

from floorflow.chip_flow.atom_health import atom_health, prefix_health


def test_prefix_health_reports_active_and_absent_atoms() -> None:
    model = nn.Module()
    model.branch = nn.Sequential(nn.Linear(4, 4), nn.SiLU(), nn.Linear(4, 2))
    loss = model.branch(torch.randn(3, 4)).square().mean()
    loss.backward()
    active = prefix_health(model, ("branch.",))
    assert active["parameter_count"] > 0
    assert active["gradient_tensor_count"] > 0
    assert active["active"]
    missing = prefix_health(model, ("missing.",))
    assert missing["tensor_count"] == 0
    assert not missing["active"]


def test_atom_health_keeps_atoms_separate() -> None:
    model = nn.Module()
    model.left = nn.Linear(3, 2)
    model.right = nn.Linear(2, 1)
    model.left(torch.ones(1, 3)).sum().backward()
    health = atom_health(
        model,
        {"left": ("left.",), "right": ("right.",)},
    )
    assert health["left"]["gradient_tensor_count"] == 2
    assert health["right"]["gradient_tensor_count"] == 0
