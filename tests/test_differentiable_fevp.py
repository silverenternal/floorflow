"""Directional tests for differentiable FEVP training proxies."""

from __future__ import annotations

import torch

from floorflow.objectives.differentiable_fevp import (
    differentiable_fevp_terms,
    soft_opening_attachment,
    soft_outside_preservation,
    soft_room_overlap,
    soft_wall_continuity,
)


def test_soft_overlap_is_monotone_and_differentiable() -> None:
    footprint = torch.ones(1, 8, 8)
    low_rooms = torch.tensor(
        [[[[0.5] * 8] * 8, [[0.25] * 8] * 8]],
        requires_grad=True,
    )
    high_rooms = torch.tensor(
        [[[[0.5] * 8] * 8, [[0.75] * 8] * 8]],
        requires_grad=True,
    )
    low = soft_room_overlap(low_rooms, footprint)
    high = soft_room_overlap(high_rooms, footprint)
    assert high > low
    high.backward()
    assert high_rooms.grad is not None


def test_wall_and_opening_terms_detect_violations() -> None:
    room = torch.zeros(1, 16, 16)
    room[:, :, 8:] = 1.0
    wall = torch.zeros(1, 16, 16)
    wall[:, :, 7:9] = 1.0
    footprint = torch.ones(1, 16, 16)
    continuum = soft_wall_continuity(room, wall, footprint)
    wall[:, :, 7:9] = 0.0
    broken = soft_wall_continuity(room, wall, footprint)
    assert broken > continuum
    opening = torch.zeros(1, 16, 16)
    opening[:, 2:4, 2:4] = 1.0
    detached = soft_opening_attachment(opening, wall, footprint)
    opacity = torch.zeros_like(opening)
    opacity[:, 7:9, 7:9] = 1.0
    attached = soft_opening_attachment(opacity, torch.ones_like(wall), footprint)
    assert detached > attached


def test_outside_preservation_and_combined_terms_are_finite() -> None:
    before = torch.zeros(1, 2, 16, 16)
    before[:, 0] = 1.0
    after = before.clone()
    after[:, 0, :, 2] = 0.0
    footprint = torch.ones(1, 16, 16)
    edit = torch.zeros(1, 16, 16)
    edit[:, 4:, :] = 1.0
    preservation = soft_outside_preservation(after, before, footprint, edit)
    assert preservation > 0
    total, terms = differentiable_fevp_terms(
        after,
        before,
        torch.ones(1, 16, 16),
        torch.zeros(1, 16, 16),
        footprint,
        edit,
    )
    assert torch.isfinite(total)
    assert all(name.startswith("soft_") for name in terms)
