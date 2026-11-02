"""Guided multi-channel flow source built from coarse CAD controls."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.flow.core import sample_prior
from floorflow.heads.coarse_furniture_refiner import coarse_furniture_control
from floorflow.heads.coarse_structure_refiner import coarse_structure_control
from floorflow.representation.coarse_layout import coarse_room_layout_logits


def coalesce_probabilities(
    value: torch.Tensor,
    size: tuple[int, int],
) -> torch.Tensor:
    return F.interpolate(
        value,
        size=size,
        mode="bilinear",
        align_corners=False,
    )


def target_channels_from_coarse_controls(
    room_logits: torch.Tensor,
    structure: torch.Tensor,
    furniture: torch.Tensor,
    footprint: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
) -> torch.Tensor:
    size = footprint.shape[-2:]
    rooms = coalesce_probabilities(room_logits.softmax(dim=1), size)
    structure = coalesce_probabilities(structure, size)
    furniture = coalesce_probabilities(furniture, size)
    room_count = len(taxonomy.room_ids)
    wall = structure[:, room_count]
    opening_types = structure[
        :,
        room_count + 2 : room_count + 2 + len(taxonomy.opening_ids),
    ]
    furniture_types = furniture[
        :,
        room_count : room_count + len(taxonomy.furniture_ids),
    ]
    channels = torch.zeros(
        footprint.shape[0],
        taxonomy.num_target_channels,
        *size,
        device=footprint.device,
        dtype=footprint.dtype,
    )
    room_ids = torch.tensor(taxonomy.room_ids, device=footprint.device)
    opening_ids = torch.tensor(taxonomy.opening_ids, device=footprint.device)
    furniture_ids = torch.tensor(taxonomy.furniture_ids, device=footprint.device)
    channels[:, room_ids - 1] = rooms
    channels[:, taxonomy.class_id("wall") - 1] = wall
    channels[:, opening_ids - 1] = opening_types
    channels[:, furniture_ids - 1] = furniture_types
    return channels.clamp(0.0, 1.0) * footprint.bool()[:, None]


def guided_source_from_labels(
    labels: torch.Tensor,
    footprint: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
    *,
    room_size: int = 224,
    structure_size: int = 128,
    furniture_size: int = 192,
) -> torch.Tensor:
    room_logits = coarse_room_layout_logits(
        labels,
        taxonomy,
        size=room_size,
    )
    structure = coarse_structure_control(
        labels,
        footprint,
        taxonomy,
        size=structure_size,
    )
    furniture = coarse_furniture_control(
        labels,
        footprint,
        taxonomy,
        size=furniture_size,
    )
    return target_channels_from_coarse_controls(
        room_logits,
        structure,
        furniture,
        footprint,
        taxonomy,
    )


@torch.no_grad()
def sample_guided_floorplans(
    model,
    footprint: torch.Tensor,
    source: torch.Tensor,
    *,
    steps: int = 8,
    floor_ids: torch.Tensor | None = None,
    noise_scale: float = 0.0,
) -> torch.Tensor:
    model.eval()
    if footprint.ndim == 3:
        footprint_bchw = footprint[:, None]
    else:
        footprint_bchw = footprint
    mask = footprint_bchw.float()
    state = source.to(mask.dtype) * mask
    if noise_scale > 0:
        state = state + sample_prior(
            footprint_bchw[:, 0],
            state.shape[1],
            noise_scale=noise_scale,
        )
    steps = max(1, int(steps))
    dt = 1.0 / steps
    for step in range(steps):
        time_value = torch.full(
            (footprint_bchw.shape[0],),
            step * dt,
            device=footprint_bchw.device,
            dtype=footprint_bchw.dtype,
        )
        state = (state + dt * model(state, time_value, footprint_bchw, floor_ids=floor_ids)) * mask
    return state


__all__ = [
    "guided_source_from_labels",
    "sample_guided_floorplans",
    "target_channels_from_coarse_controls",
]
