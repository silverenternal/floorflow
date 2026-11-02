"""Differentiable semantic statistics for style-conditioned supervision."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from floorflow.distribution.hierarchy import ChannelHierarchy
from floorflow.distribution.style_conditioning import (
    model_style_contrast_loss,
    style_orthogonality_loss,
)


def distribution_style_vector(
    pixel: dict[str, torch.Tensor],
    elements: dict[str, dict[str, torch.Tensor]],
    hierarchy: ChannelHierarchy,
    footprint: torch.Tensor,
) -> torch.Tensor:
    """Return room area/presence, family density and query statistics."""

    probability = pixel["probability"]
    if footprint.ndim == 3:
        footprint = footprint[:, None]
    if footprint.shape[-2:] != probability.shape[-2:]:
        footprint = F.interpolate(
            footprint.float(),
            size=probability.shape[-2:],
            mode="nearest",
        )
    inside = footprint.float()
    groups = hierarchy.groups
    room_indices = groups["room"]
    room_probability = probability[:, room_indices] * inside
    room_mass = room_probability.sum(dim=(2, 3))
    area_distribution = room_mass / room_mass.sum(dim=1, keepdim=True).clamp_min(
        1.0e-6
    )
    room_presence = room_probability.amax(dim=(2, 3))
    family_density = []
    for name in ("wall", "opening", "furniture", "room"):
        indices = groups[name]
        family_density.append(
            (probability[:, indices] * inside).sum(dim=(1, 2, 3))
            / inside.sum(dim=(1, 2, 3)).clamp_min(1.0)
        )
    family_density = torch.stack(family_density, dim=-1)
    room_elements = elements["room"]
    element_presence = room_elements["presence_logits"].sigmoid()
    element_count = element_presence.sum(dim=1, keepdim=True)
    element_area = (
        room_elements["mask_logits"].sigmoid().mean(dim=(2, 3))
        * element_presence
    ).sum(dim=1, keepdim=True)
    element_area = element_area * probability.shape[-2] * probability.shape[-1]
    return torch.cat(
        [
            area_distribution,
            room_presence,
            family_density,
            element_count,
            element_area,
        ],
        dim=-1,
    )


def load_style_semantic_prototypes(
    path: str | Path,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    width = 54
    return {
        "center": torch.tensor(
            payload["feature_center"][:width],
            device=device,
            dtype=torch.float32,
        ),
        "scale": torch.tensor(
            payload["feature_scale"][:width],
            device=device,
            dtype=torch.float32,
        ),
        "centroids": torch.tensor(
            [centroid[:width] for centroid in payload["centroids_standardized"]],
            device=device,
            dtype=torch.float32,
        ),
    }


def style_prototype_loss(
    vectors: torch.Tensor,
    style_ids: torch.Tensor,
    prototypes: dict[str, torch.Tensor],
) -> torch.Tensor:
    normalized = (vectors - prototypes["center"]) / prototypes["scale"]
    target = prototypes["centroids"][style_ids.long()]
    return F.smooth_l1_loss(normalized, target)


def auxiliary_style_losses(
    model,
    distribution: dict[str, object],
    hierarchy: ChannelHierarchy,
    footprint: torch.Tensor,
    style_ids: torch.Tensor | None,
    *,
    prototypes: dict[str, torch.Tensor] | None = None,
    contrast_weight: float = 0.0,
    contrast_margin: float = 0.01,
    semantic_weight: float = 0.0,
    orthogonality_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute optional style separation and semantic-prototype objectives."""

    reference = distribution["features"]
    total = reference.sum() * 0.0
    components: dict[str, torch.Tensor] = {}
    if style_ids is None:
        return total, components
    if contrast_weight > 0:
        value = model_style_contrast_loss(
            model,
            reference,
            style_ids,
            margin=contrast_margin,
        )
        total = total + float(contrast_weight) * value
        components["style_contrast_loss"] = value.detach()
    if prototypes is not None and semantic_weight > 0:
        vector = distribution_style_vector(
            distribution["pixel"],
            distribution["elements"],
            hierarchy,
            footprint,
        )
        value = style_prototype_loss(vector, style_ids, prototypes)
        total = total + float(semantic_weight) * value
        components["style_semantic_loss"] = value.detach()
    if orthogonality_weight > 0:
        value = style_orthogonality_loss(model)
        total = total + float(orthogonality_weight) * value
        components["style_orthogonality_loss"] = value.detach()
    return total, components


__all__ = [
    "auxiliary_style_losses",
    "distribution_style_vector",
    "load_style_semantic_prototypes",
    "style_prototype_loss",
]
