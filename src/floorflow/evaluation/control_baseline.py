"""Reconstruct floorplan layers directly from serialized control bundles."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def resize_nearest(value: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    input_value = value[None] if value.ndim == 3 else value[None, None]
    output = F.interpolate(input_value.float(), size=size, mode="nearest")
    return output[0] if value.ndim == 3 else output[0, 0]


def resize_logits(value: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(
        value[None].float(),
        size=size,
        mode="bilinear",
        align_corners=False,
    )[0]


def control_layers(
    controls: dict[str, torch.Tensor],
    footprint: torch.Tensor,
    taxonomy,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    size = tuple(footprint.shape[-2:])
    room_ids = torch.tensor(taxonomy.room_ids, device=footprint.device)
    opening_ids = torch.tensor(taxonomy.opening_ids, device=footprint.device)
    furniture_ids = torch.tensor(taxonomy.furniture_ids, device=footprint.device)
    room_count = len(taxonomy.room_ids)

    logits = resize_logits(controls["room_logits"], size)
    occupied = logits.max(dim=0).values > 0.0
    rooms = room_ids[logits.argmax(dim=0)] * occupied.long() * footprint.long()

    structure = controls["structure"].float()
    wall = resize_nearest(structure[room_count], size) >= 0.5
    opening_gate = resize_nearest(structure[room_count + 1], size) >= 0.5
    opening_type = resize_nearest(
        structure[room_count + 2 : room_count + 2 + len(taxonomy.opening_ids)],
        size,
    )
    openings = opening_ids[opening_type.argmax(dim=0)]
    openings = torch.where(
        opening_gate,
        openings,
        torch.zeros_like(openings),
    ) * footprint.long()

    furniture_type = resize_nearest(
        controls["furniture"].float()[room_count:-1],
        size,
    )
    furniture_gate = furniture_type.max(dim=0).values >= 0.5
    furniture = furniture_ids[furniture_type.argmax(dim=0)]
    furniture = torch.where(
        furniture_gate,
        furniture,
        torch.zeros_like(furniture),
    ) * footprint.long()
    wall = wall & footprint.bool() & (openings == 0)
    return rooms, wall.long(), openings, furniture


def control_layers_batch(
    controls: dict[str, torch.Tensor],
    footprint: torch.Tensor,
    taxonomy,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reconstruct a batch that shares a taxonomy but not a crop geometry."""

    layers = []
    for index in range(footprint.shape[0]):
        sample_footprint = footprint[index]
        if sample_footprint.ndim == 3:
            sample_footprint = sample_footprint[0]
        layers.append(
            control_layers(
                {key: value[index] for key, value in controls.items()},
                sample_footprint,
                taxonomy,
            )
        )
    return tuple(
        torch.stack([layer[channel] for layer in layers])
        for channel in range(4)
    )


__all__ = [
    "control_layers",
    "control_layers_batch",
    "resize_logits",
    "resize_nearest",
]
