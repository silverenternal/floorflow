"""Deterministic composition of pixel distributions and element masks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.distribution.hierarchy import ChannelHierarchy
from floorflow.representation.layered_floorplan import LayeredFloorplan


@dataclass(frozen=True)
class DistributionOverlaySettings:
    room_logit_weight: float = 0.0
    room_support_threshold: float = 0.1
    wall_threshold: float = 0.3
    opening_threshold: float = 0.2
    furniture_threshold: float = 0.2
    wall_element_weight: float = 0.0
    opening_element_weight: float = 0.5
    furniture_element_weight: float = 0.0

    def __post_init__(self) -> None:
        for value in (
            self.room_support_threshold,
            self.wall_threshold,
            self.opening_threshold,
            self.furniture_threshold,
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError("overlay thresholds must be in [0, 1]")
        if self.room_logit_weight < 0.0:
            raise ValueError("room_logit_weight must be non-negative")
        for value in (
            self.wall_element_weight,
            self.opening_element_weight,
            self.furniture_element_weight,
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError("element fusion weights must be in [0, 1]")


def element_family_scores(
    elements: Mapping[str, Mapping[str, torch.Tensor]],
    hierarchy: ChannelHierarchy,
    size: tuple[int, int],
) -> dict[str, torch.Tensor]:
    """Return per-class probabilities by max-marginalizing query instances."""

    scores = {}
    for name, indices in hierarchy.groups.items():
        if not indices or name not in elements:
            continue
        prediction = elements[name]
        probability = prediction["mask_logits"].sigmoid()
        type_probability = prediction["type_logits"].softmax(dim=-1)
        class_probability = type_probability[..., : len(indices)]
        query_scores = probability[:, :, None] * class_probability[:, :, :, None, None]
        class_scores = query_scores.amax(dim=1)
        scores[name] = F.interpolate(
            class_scores.float(),
            size=size,
            mode="bilinear",
            align_corners=False,
        )
    return scores


def _score_labels(
    scores: torch.Tensor,
    class_ids: tuple[int, ...],
    threshold: float,
    footprint: torch.Tensor,
) -> torch.Tensor:
    ids = torch.tensor(class_ids, device=scores.device, dtype=torch.long)
    confidence, index = scores.max(dim=1)
    labels = ids[index]
    return torch.where(
        confidence >= float(threshold),
        labels,
        torch.zeros_like(labels),
    ) * footprint.long()


def _pixel_class_scores(
    labels: torch.Tensor,
    class_ids: tuple[int, ...],
) -> torch.Tensor:
    return torch.stack(
        [(labels == class_id).float() for class_id in class_ids],
        dim=1,
    )


def overlay_room_labels(
    pixel_logits: torch.Tensor,
    room_scores: torch.Tensor,
    room_class_ids: tuple[int, ...],
    footprint: torch.Tensor,
    settings: DistributionOverlaySettings,
) -> torch.Tensor:
    room_indices = torch.tensor(
        room_class_ids,
        device=pixel_logits.device,
        dtype=torch.long,
    )
    pixel_room_logits = pixel_logits[:, room_indices]
    normalized_scores = room_scores / room_scores.sum(
        dim=1,
        keepdim=True,
    ).clamp_min(1.0e-6)
    fused = pixel_room_logits + settings.room_logit_weight * normalized_scores.clamp_min(
        1.0e-4
    ).log()
    ids = torch.tensor(room_class_ids, device=fused.device, dtype=torch.long)
    labels = ids[fused.argmax(dim=1)]
    support = room_scores.amax(dim=1) >= settings.room_support_threshold
    return labels * support.long() * footprint.long()


def compose_distribution_overlay(
    rooms: torch.Tensor,
    scores: Mapping[str, torch.Tensor],
    footprint: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
    settings: DistributionOverlaySettings,
    pixel_labels: torch.Tensor | None = None,
) -> tuple[torch.Tensor, LayeredFloorplan]:
    if pixel_labels is None:
        pixel_labels = torch.zeros_like(rooms)
    wall_id = taxonomy.class_id("wall")
    wall_score = (
        settings.wall_element_weight * scores["wall"][:, 0]
        + (1.0 - settings.wall_element_weight)
        * (pixel_labels == wall_id).float()
    )
    wall = (wall_score >= settings.wall_threshold).long() * footprint.long()
    opening_score = (
        settings.opening_element_weight * scores["opening"]
        + (1.0 - settings.opening_element_weight)
        * _pixel_class_scores(pixel_labels, taxonomy.opening_ids)
    )
    opening = _score_labels(
        opening_score,
        taxonomy.opening_ids,
        settings.opening_threshold,
        footprint,
    )
    furniture_score = (
        settings.furniture_element_weight * scores["furniture"]
        + (1.0 - settings.furniture_element_weight)
        * _pixel_class_scores(pixel_labels, taxonomy.furniture_ids)
    )
    furniture = _score_labels(
        furniture_score,
        taxonomy.furniture_ids,
        settings.furniture_threshold,
        footprint,
    )
    layers = LayeredFloorplan(
        room_labels=rooms,
        wall=wall,
        opening=opening,
        furniture=furniture,
        footprint=footprint,
        taxonomy=taxonomy,
    )
    return layers.visible_labels(), layers


__all__ = [
    "DistributionOverlaySettings",
    "compose_distribution_overlay",
    "element_family_scores",
    "overlay_room_labels",
]
