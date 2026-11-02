from __future__ import annotations

import torch

from floorflow.inference.iterative_control_sampling import sample_iterative_controlled


class DummyMaskedFlow(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward_with_aux(
        self,
        x_t,
        t,
        footprint,
        *,
        floor_ids=None,
        control,
        control_mask,
    ):
        del t, floor_ids, control_mask
        velocity = 0.1 * torch.ones_like(x_t)
        return velocity, {}


def test_iterative_control_reveals_high_confidence_regions() -> None:
    model = DummyMaskedFlow()
    footprint = torch.ones(1, 1, 8, 8)
    control = torch.zeros(1, 4, 8, 8)
    mask = torch.ones(1, 4, 8, 8, dtype=torch.bool)
    sampled, updated, remaining = sample_iterative_controlled(
        model,
        footprint,
        control,
        mask,
        rounds=2,
        steps_per_round=1,
        reveal_fraction=0.25,
    )
    assert sampled.shape == control.shape
    assert remaining.sum() < mask.sum()
    assert updated.abs().sum() > 0
