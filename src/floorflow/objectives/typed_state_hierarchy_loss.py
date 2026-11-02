"""Hierarchical primary/topology/sequence loss composition."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class HierarchyLossResult:
    total: torch.Tensor
    components: dict[str, torch.Tensor]


def typed_state_hierarchy_loss(
    terms: dict[str, torch.Tensor],
    weights: dict[str, float],
) -> HierarchyLossResult:
    groups = {
        "primary": ("polygon", "identity", "count"),
        "topology": (
            "wall",
            "opening",
            "adjacency",
            "token_structure",
        ),
        "sequence": ("token",),
    }
    components = {}
    total = None
    for group, names in groups.items():
        selected = [terms[name] for name in names if name in terms]
        group_loss = (
            torch.stack(selected).mean()
            if selected
            else next(iter(terms.values())) * 0.0
        )
        components[f"hierarchy_{group}_loss"] = group_loss.detach()
        weighted = float(weights.get(group, 1.0)) * group_loss
        total = weighted if total is None else total + weighted
    if total is None:
        raise ValueError("no hierarchy loss terms are available")
    return HierarchyLossResult(total, components)


__all__ = ["HierarchyLossResult", "typed_state_hierarchy_loss"]
