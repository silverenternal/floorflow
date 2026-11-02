import torch

from floorflow.distribution.parent_gate import ParentReliabilityGate


def test_parent_reliability_gate_is_identity_at_zero_blend() -> None:
    gate = ParentReliabilityGate(("room",))
    parent = torch.full((1, 1, 4, 4), 0.25)
    footprint = torch.ones_like(parent)
    output, diagnostics = gate("room", parent, footprint)
    assert torch.equal(output, parent)
    assert diagnostics["effect"].item() == 0.0


def test_parent_reliability_gate_can_normalize_sparse_support() -> None:
    gate = ParentReliabilityGate(("room",))
    gate.blend["room"].data.fill_(1.0)
    parent = torch.full((1, 1, 4, 4), 0.25)
    parent[:, :, :2] = 0.0
    footprint = torch.ones_like(parent)
    output, diagnostics = gate("room", parent, footprint)
    assert output[:, :, 2:].min() > 0.8
    assert torch.equal(output[:, :, :2], torch.zeros_like(output[:, :, :2]))
    assert diagnostics["coverage"] > 0
    assert diagnostics["effect"] > 0
