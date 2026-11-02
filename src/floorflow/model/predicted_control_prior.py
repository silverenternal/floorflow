"""Canonical prior from predicted-control channels to target logits."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from floorflow.data.taxonomy import FloorplanTaxonomy


def predicted_control_prior(
    taxonomy: FloorplanTaxonomy,
    controls: dict[str, torch.Tensor],
    grid: int,
) -> torch.Tensor:
    """Map predicted-control channels into the target logit vocabulary."""

    batch = controls["room_logits"].shape[0]
    size = (int(grid), int(grid))
    prior = controls["room_logits"].new_zeros(
        batch,
        taxonomy.num_classes,
        *size,
        dtype=torch.float32,
    )
    prior[:, 0] = -6.0
    room_logits = F.interpolate(
        controls["room_logits"].float(),
        size=size,
        mode="bilinear",
        align_corners=False,
    )
    room_ids = torch.tensor(taxonomy.room_ids, device=prior.device)
    prior[:, room_ids] = room_logits
    structure = F.interpolate(
        controls["structure"].float(),
        size=size,
        mode="bilinear",
        align_corners=False,
    )
    room_count = len(taxonomy.room_ids)
    prior[:, taxonomy.class_id("wall")] = torch.logit(
        structure[:, room_count].clamp(1.0e-4, 1.0 - 1.0e-4)
    )
    opening_probability = structure[:, room_count + 1].clamp(
        1.0e-4,
        1.0 - 1.0e-4,
    )
    opening_types = structure[
        :,
        room_count + 2 : room_count + 2 + len(taxonomy.opening_ids),
    ].clamp_min(1.0e-4)
    opening_ids = torch.tensor(taxonomy.opening_ids, device=prior.device)
    prior[:, opening_ids] = torch.log(
        opening_probability[:, None] * opening_types
    ).clamp(-12.0, 6.0)
    furniture = F.interpolate(
        controls["furniture"].float(),
        size=size,
        mode="bilinear",
        align_corners=False,
    )
    furniture_ids = torch.tensor(
        taxonomy.furniture_ids,
        device=prior.device,
    )
    prior[:, furniture_ids] = torch.log(
        furniture[:, : len(taxonomy.furniture_ids)].clamp_min(1.0e-4)
    ).clamp(-12.0, 6.0)
    return prior


__all__ = ["predicted_control_prior"]
