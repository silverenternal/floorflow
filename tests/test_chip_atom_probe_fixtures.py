"""Tests for reusable chip atom probe fixtures."""

from __future__ import annotations

import torch

from floorflow.chip_flow.atom_probe_fixtures import (
    ATOM_PREFIXES,
    activate_probe_parameters,
    full_probe_model,
    probe_args,
    probe_batch,
)


def test_probe_batch_and_args_expose_required_contracts() -> None:
    batch = probe_batch(batch=2, count=8, pins=12)
    assert batch["features"].shape == (2, 8, 19)
    assert batch["pins"].shape == (2, 12, 2)
    assert batch["net_block_incidence"].shape == (2, 8, 8)
    assert probe_args().hierarchy_port_weight == 0.1


def test_probe_fixture_activates_scalar_gates_and_is_reusable() -> None:
    model = full_probe_model()
    activate_probe_parameters(model)
    gates = [
        parameter
        for name, parameter in model.named_parameters()
        if name.endswith("gate") and parameter.numel() == 1
    ]
    assert gates
    assert any(torch.allclose(gate, torch.tensor(0.5)) for gate in gates)
    assert "A29" in ATOM_PREFIXES
