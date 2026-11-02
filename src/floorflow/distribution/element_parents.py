"""Hierarchical parent gates shared by the pixel and element heads."""

from __future__ import annotations

from collections.abc import Mapping

import torch


def parent_map(
    name: str,
    family_gates: Mapping[str, torch.Tensor],
    reference: torch.Tensor,
    *,
    room_parent_gate: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the spatial parent gate for an element family."""

    if name == "wall":
        return torch.ones_like(reference[:, :1])
    if name == "opening":
        return family_gates.get("wall", torch.ones_like(reference[:, :1]))
    if name == "room":
        if room_parent_gate is not None:
            return room_parent_gate
        wall = family_gates.get("wall", torch.zeros_like(reference[:, :1]))
        return (1.0 - wall).clamp(0.0, 1.0)
    return family_gates.get("room", torch.ones_like(reference[:, :1]))


def family_parent_gates(
    family_gates: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Explicit pixel → element parent gates for diagnostics and routing."""

    gates: dict[str, torch.Tensor] = {}
    wall = family_gates.get("wall")
    room = family_gates.get("room")
    if wall is not None:
        gates["wall"] = wall
        gates["opening"] = wall
        gates["room_region"] = (1.0 - wall).clamp(0.0, 1.0)
    if room is not None:
        gates["room"] = room
        gates["furniture"] = room
        gates["support"] = room
    return gates


__all__ = ["family_parent_gates", "parent_map"]
