"""Dynamic masking of coarse CAD controls for guided flow matching."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from floorflow.data.taxonomy import FloorplanTaxonomy


@dataclass(frozen=True)
class ControlMask:
    value: torch.Tensor
    spatial: torch.Tensor
    group: torch.Tensor
    ratio: torch.Tensor


def control_channel_groups(taxonomy: FloorplanTaxonomy) -> dict[str, torch.Tensor]:
    room_ids = torch.tensor(taxonomy.room_ids, dtype=torch.long)
    opening_ids = torch.tensor(taxonomy.opening_ids, dtype=torch.long)
    furniture_ids = torch.tensor(taxonomy.furniture_ids, dtype=torch.long)
    return {
        "room": room_ids - 1,
        "wall": torch.tensor([taxonomy.class_id("wall") - 1]),
        "opening": opening_ids - 1,
        "furniture": furniture_ids - 1,
    }


def sample_spatial_mask(
    source: torch.Tensor,
    footprint: torch.Tensor,
    ratio_tensor: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if footprint.ndim == 3:
        footprint = footprint[:, None]
    batch, _channels, height, width = source.shape
    device = source.device
    spatial = torch.zeros(
        batch,
        1,
        height,
        width,
        device=device,
        dtype=torch.bool,
    )
    for index in range(batch):
        fraction = float(ratio_tensor[index].clamp(0.0, 1.0))
        if fraction <= 0.0:
            continue
        if fraction >= 1.0:
            spatial[index, 0] = footprint[index, 0].bool()
            continue
        side = max(1, int(round((height * width * fraction) ** 0.5)))
        block_h = min(height, side)
        block_w = min(width, side)
        for _attempt in range(3):
            top = int(
                torch.randint(
                    0,
                    max(1, height - block_h + 1),
                    (1,),
                    device=device,
                    generator=generator,
                )
            )
            left = int(
                torch.randint(
                    0,
                    max(1, width - block_w + 1),
                    (1,),
                    device=device,
                    generator=generator,
                )
            )
            spatial[index, 0, top : top + block_h, left : left + block_w] = True
        spatial[index, 0] &= footprint[index, 0].bool()
    return spatial


def sample_control_mask(
    source: torch.Tensor,
    footprint: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
    *,
    ratio: float | None = None,
    generator: torch.Generator | None = None,
) -> ControlMask:
    if source.ndim != 4:
        raise ValueError("source must be [B,C,H,W]")
    if footprint.ndim == 3:
        footprint = footprint[:, None]
    batch, _channels, height, width = source.shape
    device = source.device
    if ratio is None:
        ratio_tensor = torch.empty(batch, device=device).uniform_(
            0.15,
            0.9,
            generator=generator,
        )
    else:
        ratio_tensor = torch.full(
            (batch,),
            float(ratio),
            device=device,
            dtype=source.dtype,
        )
    spatial = sample_spatial_mask(
        source,
        footprint,
        ratio_tensor,
        generator=generator,
    )
    groups = control_channel_groups(taxonomy)
    group = torch.zeros(
        batch,
        source.shape[1],
        device=device,
        dtype=torch.bool,
    )
    group_names = tuple(groups)
    if ratio is None or float(ratio) > 0.0:
        for index in range(batch):
            order = torch.randperm(
                len(group_names),
                device=device,
                generator=generator,
            )
            count = max(
                1,
                int(round(len(group_names) * float(ratio_tensor[index]))),
            )
            for selected in order[:count].tolist():
                group[index, groups[group_names[selected]].to(device)] = True
    group = group[:, :, None, None].expand(
        batch,
        source.shape[1],
        height,
        width,
    )
    value = (spatial.expand_as(group) | group) & footprint.bool()
    return ControlMask(
        value=value,
        spatial=spatial,
        group=group,
        ratio=ratio_tensor,
    )


def apply_control_mask(
    source: torch.Tensor,
    mask: torch.Tensor,
    missing_token: torch.Tensor,
) -> torch.Tensor:
    if missing_token.ndim != 1 or missing_token.shape[0] != source.shape[1]:
        raise ValueError("missing_token must match source channels")
    token = missing_token.to(source.dtype)[None, :, None, None]
    return torch.where(mask, token, source)


__all__ = [
    "ControlMask",
    "apply_control_mask",
    "control_channel_groups",
    "sample_control_mask",
    "sample_spatial_mask",
]
