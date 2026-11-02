"""Complexity-conditioned semantic element query budgets."""

from __future__ import annotations

from collections.abc import Mapping

import torch


def allocate_element_query_budgets(
    family_gates: Mapping[str, torch.Tensor],
    reference: torch.Tensor,
    query_counts: Mapping[str, int],
    *,
    class_density: Mapping[str, torch.Tensor] | None = None,
    recycle_strength: float = 0.5,
    active_threshold: float = 0.01,
) -> dict[str, torch.Tensor]:
    """Allocate per-sample queries from family density and rare-class coverage.

    ``class_density`` maps a family to ``[B,K]`` per-class probability mass.
    Families whose predicted coverage is concentrated on a few classes get a
    recycling boost, and the budget is never below the number of active
    classes, so rare furniture anchors keep a query slot.
    """

    room_mass = family_gates.get(
        "room",
        torch.ones_like(reference[:, :1]),
    ).flatten(1).mean(dim=1)
    minimum = {
        "wall": 2,
        "opening": 2,
        "room": 3,
        "furniture": 2,
        "support": 1,
    }
    budgets = {}
    for name, count in query_counts.items():
        detail = family_gates.get(
            name,
            torch.zeros_like(reference[:, :1]),
        ).flatten(1).mean(dim=1)
        score = (0.5 * room_mass + detail).clamp(0.0, 1.0)
        low = min(int(minimum[name]), int(count))
        if class_density is not None and name in class_density:
            density = class_density[name].to(reference.device).clamp_min(0.0)
            active = (density >= float(active_threshold)).sum(dim=1).float()
            mass = density.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
            concentration = (density / mass).amax(dim=1).clamp(0.0, 1.0)
            score = (
                score
                + float(recycle_strength) * concentration
            ).clamp(0.0, 1.0)
            low = torch.maximum(
                torch.full_like(active, low),
                active.clamp_max(float(count)),
            )
        else:
            active = None
        headroom = int(count) - low
        if torch.is_tensor(headroom):
            headroom = headroom.clamp_min(0.0)
        else:
            headroom = max(0, int(headroom))
        value = torch.ceil(
            low + score * headroom
        ).long().clamp_max(int(count))
        if torch.is_tensor(low):
            value = torch.maximum(value, low.to(value.dtype).long())
        else:
            value = value.clamp_min(int(low))
        budgets[name] = value
    return budgets


__all__ = ["allocate_element_query_budgets"]
