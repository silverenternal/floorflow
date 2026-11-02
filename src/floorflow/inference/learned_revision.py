"""Consume learned revision state in the typed edit graph."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from floorflow.inference.typed_edit_graph import (
    FAMILIES,
    TypedEditGraph,
    build_typed_edit_graph,
)


@torch.no_grad()
def revision_family_names(
    family_logits: torch.Tensor,
    *,
    threshold: float = 0.5,
) -> tuple[str, ...]:
    probability = family_logits.detach().float().sigmoid()[0]
    return tuple(
        name
        for name, value in zip(FAMILIES, probability, strict=True)
        if float(value) >= float(threshold)
    )


def revision_gate_value(
    revision: dict[str, torch.Tensor],
) -> float:
    """Return one validated scalar revision-admission probability."""

    gate = revision.get("revision_gate")
    if gate is None:
        return 1.0
    if gate.numel() != 1 or not torch.isfinite(gate).all():
        raise ValueError("revision_gate must be one finite scalar")
    value = float(gate.detach().float().reshape(-1)[0])
    if not 0.0 <= value <= 1.0:
        raise ValueError("revision_gate must be a probability")
    return value


@torch.no_grad()
def consume_learned_revision(
    before_labels: np.ndarray,
    edit_region: np.ndarray,
    changed_mask: np.ndarray,
    revision: dict[str, torch.Tensor],
    *,
    room_ids: Sequence[int],
    wall_id: int,
    opening_ids: Sequence[int],
    furniture_ids: Sequence[int],
    family_threshold: float = 0.5,
    region_threshold: float = 0.5,
    gate_threshold: float = 0.5,
) -> tuple[TypedEditGraph, tuple[str, ...]]:
    """Use admitted revision state, otherwise fall back to the edit contract."""

    admitted = revision_gate_value(revision) >= float(gate_threshold)
    families = (
        revision_family_names(
            revision["family_logits"],
            threshold=family_threshold,
        )
        if admitted
        else None
    )
    if admitted:
        region = revision["region_logits"].detach().float().sigmoid()[0, 0]
        learned_region = region.cpu().numpy() >= float(region_threshold)
    else:
        learned_region = np.zeros_like(
            np.asarray(edit_region, dtype=bool),
        )
    if learned_region.shape != np.asarray(edit_region).shape:
        raise ValueError("learned region and edit region must share a shape")
    graph = build_typed_edit_graph(
        before_labels,
        np.asarray(edit_region, dtype=bool) | learned_region,
        changed_mask,
        room_ids=room_ids,
        wall_id=wall_id,
        opening_ids=opening_ids,
        furniture_ids=furniture_ids,
        changed_families=families,
    )
    return graph, graph.changed_families


__all__ = [
    "consume_learned_revision",
    "revision_family_names",
    "revision_gate_value",
]
