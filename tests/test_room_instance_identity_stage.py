from __future__ import annotations

import torch

from floorflow.model.room_instance_identity_stage import (
    owner_aware_boundary,
)


def test_owner_aware_boundary_exposes_second_owner_ambiguity() -> None:
    probability = torch.zeros(1, 3, 5, 5)
    probability[:, 0] = 0.9
    probability[:, 1] = 0.06
    probability[:, 2] = 0.04
    probability[:, :, 2, 2] = torch.tensor([0.55, 0.45, 0.0])

    boundary = owner_aware_boundary(probability)

    assert boundary.shape == (1, 1, 5, 5)
    assert boundary[0, 0, 2, 2] >= 0.45
    assert boundary[0, 0, 0, 0] <= 0.17
