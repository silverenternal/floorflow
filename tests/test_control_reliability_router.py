from __future__ import annotations

import pytest
import torch

from floorflow.inference.control_reliability_router import (
    footprint_confidence,
    label_agreement,
    route_room_control,
)


def test_footprint_confidence_ignores_outside_pixels() -> None:
    probability = torch.tensor(
        [
            [[[0.9, 0.0], [0.0, 0.0]], [[0.1, 0.0], [0.0, 0.0]]],
        ]
    )
    footprint = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])
    assert footprint_confidence(probability, footprint).item() == pytest.approx(0.9)


def test_agreement_uses_only_footprint() -> None:
    primary = torch.tensor([[[1, 1], [1, 0]]])
    candidate = torch.tensor([[[1, 0], [0, 0]]])
    footprint = torch.tensor([[[1.0, 1.0], [0.0, 0.0]]])
    assert label_agreement(primary, candidate, footprint).item() == 0.5


def test_router_requires_confidence_and_agreement() -> None:
    primary = torch.tensor(
        [
            [
                [[2.0, 2.0], [2.0, 2.0]],
                [[0.0, 0.0], [0.0, 0.0]],
            ],
            [
                [[2.0, 2.0], [2.0, 2.0]],
                [[0.0, 0.0], [0.0, 0.0]],
            ],
        ]
    )
    candidate = primary.clone()
    candidate[0, 0] += 2.0
    candidate[0, 1] -= 2.0
    candidate[1, 0] -= 2.0
    candidate[1, 1] += 2.0
    footprint = torch.ones(2, 2, 2)
    routed = route_room_control(
        primary,
        candidate,
        footprint,
        minimum_agreement=0.5,
    )
    assert routed.selected_candidate[0]
    assert not routed.selected_candidate[1]
