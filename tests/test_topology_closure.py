from __future__ import annotations

import torch

from floorflow.objectives.topology_closure import (
    low_confidence_gap_closure_loss,
)


def test_low_confidence_gap_closure_prefers_closed_room_seams() -> None:
    labels = torch.zeros((1, 16, 16), dtype=torch.long)
    labels[:, :, :8] = 2
    labels[:, :, 8:] = 3
    footprint = torch.ones((1, 1, 16, 16))
    uncertain = torch.zeros((1, 1, 16, 16))
    closed = torch.full((1, 1, 16, 16), 4.0)
    uncertain_loss = low_confidence_gap_closure_loss(
        uncertain,
        labels,
        footprint,
        (2, 3),
    )
    closed_loss = low_confidence_gap_closure_loss(
        closed,
        labels,
        footprint,
        (2, 3),
    )
    assert closed_loss < uncertain_loss
