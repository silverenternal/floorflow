"""Fuse coarse structural control with refined wall/opening outputs."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from floorflow.data.taxonomy import FloorplanTaxonomy


def opening_labels_from_control(
    control: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
    *,
    output_size: tuple[int, int],
    threshold: float = 0.2,
) -> torch.Tensor:
    room_count = len(taxonomy.room_ids)
    wall_index = room_count
    opening_index = wall_index + 1
    type_start = opening_index + 1
    type_end = type_start + len(taxonomy.opening_ids)
    opening_support = control[:, opening_index : opening_index + 1]
    opening_types = control[:, type_start:type_end]
    support = F.interpolate(
        opening_support,
        size=output_size,
        mode="nearest",
    )[:, 0] >= float(threshold)
    local_type = F.interpolate(
        opening_types,
        size=output_size,
        mode="nearest",
    ).argmax(dim=1)
    opening_ids = torch.tensor(
        taxonomy.opening_ids,
        device=control.device,
    )
    labels = opening_ids[local_type]
    return torch.where(support, labels, torch.zeros_like(labels))


__all__ = ["opening_labels_from_control"]
