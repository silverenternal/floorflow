"""Layered losses for CAD action-stage outputs."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.heads.room_mask_losses import soft_tversky_loss
from floorflow.objectives.per_case_binary import per_case_focal_loss
from floorflow.objectives.semantic_losses import dice_loss
from floorflow.objectives.structural_consistency import (
    opening_wall_affinity_loss,
)


def cad_stage_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    footprint: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
    *,
    wall_weight: float = 1.0,
    opening_weight: float = 1.0,
    furniture_weight: float = 0.5,
    wall_precision_weight: float = 0.0,
    opening_precision_weight: float = 0.0,
    opening_host_weight: float = 0.0,
    return_loss_terms: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]] | tuple[
    torch.Tensor,
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
]:
    """Supervise wall and object layers independently from room partitions."""

    if logits.shape[-2:] != target.shape[-2:]:
        logits = F.interpolate(
            logits,
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    valid = footprint.bool()
    wall_logits = logits[:, taxonomy.class_id("wall")]
    wall_target = (target == taxonomy.class_id("wall")) & valid
    wall_geometry_loss = per_case_focal_loss(
        wall_logits,
        wall_target.float(),
        valid,
    )
    wall_geometry_loss = wall_geometry_loss + dice_loss(
        wall_logits.masked_fill(~valid, -20.0),
        wall_target.float(),
    )
    wall_precision_loss = soft_tversky_loss(
        wall_logits.masked_fill(~valid, -20.0),
        wall_target.float(),
        alpha=0.7,
        beta=0.3,
    )
    wall_loss = (
        wall_geometry_loss
        + float(wall_precision_weight) * wall_precision_loss
    )

    opening_ids = torch.tensor(taxonomy.opening_ids, device=logits.device)
    opening_logits = logits[:, opening_ids]
    opening_foreground = torch.logsumexp(opening_logits, dim=1) - 0.5 * torch.log(
        opening_logits.new_tensor(float(len(opening_ids)))
    )
    opening_target = torch.isin(target, opening_ids) & valid
    opening_geometry_loss = per_case_focal_loss(
        opening_foreground,
        opening_target.float(),
        valid,
    )
    opening_geometry_loss = opening_geometry_loss + dice_loss(
        opening_foreground.masked_fill(~valid, -20.0),
        opening_target.float(),
    )
    opening_precision_loss = soft_tversky_loss(
        opening_foreground.masked_fill(~valid, -20.0),
        opening_target.float(),
        alpha=0.7,
        beta=0.3,
    )
    opening_background = torch.full_like(
        opening_foreground,
        -0.5 * float(len(opening_ids)),
    )
    opening_host_loss = opening_wall_affinity_loss(
        opening_logits,
        opening_background,
        target,
        footprint,
        taxonomy,
    )
    type_target = torch.full_like(target, -1)
    for index, class_id in enumerate(taxonomy.opening_ids):
        type_target[target == class_id] = index
    type_logits = opening_logits.permute(0, 2, 3, 1)
    per_case_type_losses = [
        F.cross_entropy(
            type_logits[index][opening_target[index]],
            type_target[index][opening_target[index]],
        )
        for index in range(target.shape[0])
        if bool(opening_target[index].any())
    ]
    type_loss = (
        torch.stack(per_case_type_losses).mean()
        if per_case_type_losses
        else opening_foreground.sum() * 0.0
    )
    opening_loss = (
        opening_geometry_loss
        + float(opening_precision_weight) * opening_precision_loss
        + float(opening_host_weight) * opening_host_loss
        + 0.5 * type_loss
    )

    furniture_ids = torch.tensor(taxonomy.furniture_ids, device=logits.device)
    furniture_logits = logits[:, furniture_ids]
    furniture_foreground = torch.logsumexp(furniture_logits, dim=1)
    furniture_target = torch.isin(target, furniture_ids) & valid
    furniture_loss = per_case_focal_loss(
        furniture_foreground,
        furniture_target.float(),
        valid,
    )
    furniture_loss = furniture_loss + dice_loss(
        furniture_foreground.masked_fill(~valid, -20.0),
        furniture_target.float(),
    )
    total = (
        float(wall_weight) * wall_loss
        + float(opening_weight) * opening_loss
        + float(furniture_weight) * furniture_loss
    )
    components = {
        "stage_wall_loss": wall_loss.detach(),
        "stage_wall_precision_loss": wall_precision_loss.detach(),
        "stage_opening_loss": opening_loss.detach(),
        "stage_opening_precision_loss": (
            opening_precision_loss.detach()
        ),
        "stage_opening_host_loss": opening_host_loss.detach(),
        "stage_opening_type_loss": type_loss.detach(),
        "stage_furniture_loss": furniture_loss.detach(),
        "stage_family_total": total.detach(),
    }
    if return_loss_terms:
        return total, components, {
            "stage_wall": wall_geometry_loss,
            "stage_wall_precision": wall_precision_loss,
            "stage_opening": opening_geometry_loss,
            "stage_opening_precision": opening_precision_loss,
            "stage_opening_host": opening_host_loss,
            "stage_opening_type": type_loss,
            "stage_furniture": furniture_loss,
        }
    return total, components


__all__ = ["cad_stage_loss"]
