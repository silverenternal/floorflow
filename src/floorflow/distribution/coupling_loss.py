"""Consistency between semantic-element and dense pixel distributions."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F

from floorflow.distribution.hierarchy import ChannelHierarchy


def element_pixel_projection(
    element: Mapping[str, torch.Tensor],
    num_classes: int,
) -> torch.Tensor:
    masks = element["mask_logits"].sigmoid()
    type_probability = element["type_logits"].softmax(dim=-1)[
        :,
        :,
        : max(1, num_classes - 1),
    ]
    return torch.einsum(
        "bqhw,bqk->bkhw",
        masks,
        type_probability,
    )


def element_to_pixel_probability(
    element_output: Mapping[str, dict[str, torch.Tensor]],
    hierarchy: ChannelHierarchy,
    reference: torch.Tensor,
) -> torch.Tensor:
    probability = reference.new_zeros(
        reference.shape[0],
        hierarchy.num_classes - 1,
        *reference.shape[-2:],
    )
    for name, indices in hierarchy.groups.items():
        if not indices or name not in element_output:
            continue
        projected = element_pixel_projection(
            element_output[name],
            len(indices) + 1,
        )
        if projected.shape[-2:] != probability.shape[-2:]:
            projected = F.interpolate(
                projected,
                size=probability.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        foreground = hierarchy.foreground_indices(indices)
        probability[:, foreground] = projected.to(probability.dtype)
    return probability


def parent_consistency_loss(
    pixel_output: dict[str, torch.Tensor],
    element_output: Mapping[str, dict[str, torch.Tensor]],
    hierarchy: ChannelHierarchy,
    footprint: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    groups = hierarchy.groups
    wall_probability = pixel_output["probability"][:, groups["wall"]].sum(
        dim=1,
        keepdim=True,
    )
    room_probability = pixel_output["probability"][:, groups["room"]].sum(
        dim=1,
        keepdim=True,
    )
    wall_neighborhood = F.max_pool2d(
        wall_probability,
        kernel_size=5,
        stride=1,
        padding=2,
    )
    opening_mass = (
        element_output["opening"]["mask_logits"].sigmoid().amax(dim=1, keepdim=True)
        if "opening" in element_output
        else pixel_output["probability"][:, :1] * 0.0
    )
    furniture_mass = (
        element_output["furniture"]["mask_logits"].sigmoid().amax(dim=1, keepdim=True)
        if "furniture" in element_output
        else pixel_output["probability"][:, :1] * 0.0
    )
    inside = footprint.bool()
    if inside.shape[-2:] != opening_mass.shape[-2:]:
        inside = F.interpolate(
            inside.float(),
            size=opening_mass.shape[-2:],
            mode="nearest",
        ).bool()
    wall_small = F.interpolate(
        wall_neighborhood,
        size=opening_mass.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    room_small = F.interpolate(
        room_probability,
        size=furniture_mass.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    opening_penalty = (
        opening_mass * (1.0 - wall_small) * inside
    ).sum() / opening_mass.sum().clamp_min(1.0)
    furniture_penalty = (
        furniture_mass * (1.0 - room_small) * inside
    ).sum() / furniture_mass.sum().clamp_min(1.0)
    total = opening_penalty + furniture_penalty
    return total, {
        "parent_opening_wall_loss": opening_penalty.detach(),
        "parent_furniture_room_loss": furniture_penalty.detach(),
        "parent_consistency_loss": total.detach(),
    }


def distribution_coupling_loss(
    pixel_output: dict[str, torch.Tensor],
    element_output: Mapping[str, dict[str, torch.Tensor]],
    hierarchy: ChannelHierarchy,
    footprint: torch.Tensor | None = None,
    parent_weight: float = 0.1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if footprint is None:
        footprint = torch.ones_like(pixel_output["logits"][:, :1])
    elif footprint.ndim == 3:
        footprint = footprint[:, None]
    valid = footprint.bool().expand(-1, -1, *pixel_output["logits"].shape[-2:])
    total = pixel_output["logits"].sum() * 0.0
    components: dict[str, torch.Tensor] = {}
    full_element_probability = pixel_output.get(
        "element_feedback_probability"
    )
    groups = hierarchy.groups
    for name in ("wall", "opening", "room", "furniture", "support"):
        indices = groups[name]
        if not indices or name not in element_output:
            continue
        pixel_probability = pixel_output["probability"][:, indices]
        pixel_probability = pixel_probability / pixel_probability.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1.0e-6)
        projected = (
            full_element_probability[
                :,
                hierarchy.foreground_indices(indices),
            ]
            if full_element_probability is not None
            else element_pixel_projection(
                element_output[name],
                len(indices) + 1,
            )
        )
        if projected.shape[-2:] != pixel_probability.shape[-2:]:
            projected = F.interpolate(
                projected,
                size=pixel_probability.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        projected = projected / projected.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1.0e-6)
        kl = F.kl_div(
            projected.clamp_min(1.0e-8).log(),
            pixel_probability,
            reduction="none",
        )
        kl = (kl * valid).sum() / valid.sum().clamp_min(1)
        intersection = (projected * pixel_probability * valid).sum(dim=(2, 3))
        denominator = (
            (projected + pixel_probability) * valid
        ).sum(dim=(2, 3))
        dice = 1.0 - (
            (2.0 * intersection + 1.0)
            / (denominator + 1.0)
        ).mean()
        group_loss = kl + dice
        total = total + group_loss
        components[f"coupling_{name}_kl"] = kl.detach()
        components[f"coupling_{name}_dice"] = dice.detach()
        components[f"coupling_{name}_loss"] = group_loss.detach()
    parent_loss, parent_components = parent_consistency_loss(
        pixel_output,
        element_output,
        hierarchy,
        footprint,
    )
    components.update(parent_components)
    total = total + float(parent_weight) * parent_loss
    components["distribution_coupling_loss"] = total.detach()
    return total, components


__all__ = [
    "distribution_coupling_loss",
    "element_pixel_projection",
    "element_to_pixel_probability",
    "parent_consistency_loss",
]
