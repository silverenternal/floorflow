import torch

from floorflow.distribution.support_gate import (
    apply_support_gate,
    parent_support_band,
)


def test_parent_support_band_is_bounded_and_dilated() -> None:
    parent = torch.zeros(1, 1, 9, 9)
    parent[:, :, 4, 4] = 1.0
    footprint = torch.ones_like(parent)
    band = parent_support_band(parent, footprint, radius=1)
    assert float(band.min()) == 0.0
    assert float(band.max()) == 1.0
    assert float(band[:, :, 3:6, 3:6].min()) == 1.0


def test_support_gate_only_suppresses_outside_parent_band() -> None:
    logits = torch.zeros(1, 2, 9, 9)
    parent = torch.zeros(1, 1, 9, 9)
    parent[:, :, 4, 4] = 1.0
    footprint = torch.ones_like(parent)
    gated = apply_support_gate(
        logits,
        parent,
        footprint,
        strength=2.0,
        radius=1,
    )
    assert torch.equal(gated[:, :, 3:6, 3:6], logits[:, :, 3:6, 3:6])
    assert torch.all(gated[:, :, 0, 0] == -2.0)
