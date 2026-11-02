"""Contour-only recursive partition proposals and seam-safe label merging."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.inference.room_repair import merge_room_fragments


@dataclass(frozen=True)
class PartitionRegion:
    index: int
    depth: int
    parent_index: int
    bbox: tuple[int, int, int, int]
    mask: torch.Tensor
    recursion_strength: float

    @property
    def area(self) -> int:
        return int(self.mask.sum())


def _bbox(mask: torch.Tensor) -> tuple[int, int, int, int]:
    indices = mask.nonzero(as_tuple=False)
    if indices.numel() == 0:
        return (0, 0, 0, 0)
    top = int(indices[:, 0].min())
    left = int(indices[:, 1].min())
    bottom = int(indices[:, 0].max()) + 1
    right = int(indices[:, 1].max()) + 1
    return top, left, bottom, right


def _candidate_split(
    mask: torch.Tensor,
    *,
    min_pixels: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    bbox = _bbox(mask)
    top, left, bottom, right = bbox
    crop = mask[top:bottom, left:right]
    height, width = crop.shape
    axis = 0 if height >= width else 1
    projection = crop.sum(dim=1 if axis == 0 else 0)
    occupied = projection > 0
    positions = occupied.nonzero(as_tuple=False).flatten()
    if positions.numel() < 2:
        return None
    cumulative = projection.cumsum(dim=0)
    total = cumulative[-1]
    split_local = int(torch.searchsorted(cumulative, total * 0.5).item())
    split_local = max(int(positions[0]) + 1, min(int(positions[-1]), split_local + 1))
    first = torch.zeros_like(mask)
    second = torch.zeros_like(mask)
    if axis == 0:
        first[top : top + split_local] = mask[top : top + split_local]
        second[top + split_local : bottom] = mask[top + split_local : bottom]
    else:
        first[:, left : left + split_local] = mask[:, left : left + split_local]
        second[:, left + split_local : right] = mask[:, left + split_local : right]
    if first.sum() < min_pixels or second.sum() < min_pixels:
        return None
    return first, second


def propose_partition(
    footprint: torch.Tensor,
    recursion_strength: float,
    *,
    max_depth: int = 3,
    min_leaf_pixels: int = 256,
) -> list[PartitionRegion]:
    """Build a contour-only binary partition tree from predicted recursion need."""

    if footprint.ndim == 3:
        footprint = footprint[0]
    mask = footprint.bool().cpu()
    if not mask.any():
        raise ValueError("footprint must contain at least one pixel")
    strength = min(1.0, max(0.0, float(recursion_strength)))
    depth_budget = round(strength * max(0, int(max_depth)))
    pending = [(0, -1, 0, mask)]
    regions: list[PartitionRegion] = []
    next_index = 0
    while pending:
        index, parent, depth, region_mask = pending.pop(0)
        can_split = (
            depth < depth_budget
            and region_mask.sum() >= 2 * min_leaf_pixels
        )
        children = (
            _candidate_split(region_mask, min_pixels=min_leaf_pixels)
            if can_split
            else None
        )
        if children is None:
            regions.append(
                PartitionRegion(
                    index=index,
                    depth=depth,
                    parent_index=parent,
                    bbox=_bbox(region_mask),
                    mask=region_mask,
                    recursion_strength=strength,
                )
            )
            continue
        first, second = children
        next_index = max(next_index, index + 1)
        left_index = next_index
        right_index = next_index + 1
        next_index += 2
        pending.append((left_index, index, depth + 1, first))
        pending.append((right_index, index, depth + 1, second))
    regions.sort(key=lambda region: region.index)
    return regions


def propose_batch_partitions(
    footprints: torch.Tensor,
    recursion_strengths: torch.Tensor,
    *,
    max_depth: int = 3,
    min_leaf_pixels: int = 256,
) -> list[list[PartitionRegion]]:
    if footprints.shape[0] != recursion_strengths.shape[0]:
        raise ValueError("one recursion strength is required per footprint")
    return [
        propose_partition(
            footprints[index],
            float(recursion_strengths[index]),
            max_depth=max_depth,
            min_leaf_pixels=min_leaf_pixels,
        )
        for index in range(footprints.shape[0])
    ]


def merge_region_labels(
    footprint: torch.Tensor,
    regions: Sequence[PartitionRegion],
    local_labels: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Place independently decoded region labels into one raster without overlap."""

    if footprint.ndim == 3:
        footprint = footprint[0]
    if len(regions) != len(local_labels):
        raise ValueError("one local label map is required per region")
    output = torch.zeros_like(footprint, dtype=local_labels[0].dtype)
    for region, labels in zip(regions, local_labels):
        if labels.ndim == 3:
            labels = labels[0]
        top, left, bottom, right = region.bbox
        resized = F.interpolate(
            labels[None, None].float(),
            size=(bottom - top, right - left),
            mode="nearest",
        )[0, 0].to(output.dtype)
        placement = torch.zeros_like(output)
        placement[top:bottom, left:right] = resized
        selected = placement[region.mask]
        output[region.mask] = selected
    return output * footprint.to(output.dtype)


def merge_region_logits(
    footprint: torch.Tensor,
    regions: Sequence[PartitionRegion],
    local_logits: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Merge per-region class logits; unowned pixels receive negative infinity."""

    if footprint.ndim == 3:
        footprint = footprint[0]
    if len(regions) != len(local_logits):
        raise ValueError("one local logit map is required per region")
    if not local_logits:
        raise ValueError("at least one local logit map is required")
    channels = int(local_logits[0].shape[0])
    output = torch.full(
        (channels, *footprint.shape),
        float("-inf"),
        device=footprint.device,
        dtype=local_logits[0].dtype,
    )
    for region, logits in zip(regions, local_logits):
        top, left, bottom, right = region.bbox
        resized = F.interpolate(
            logits[None].float(),
            size=(bottom - top, right - left),
            mode="bilinear",
            align_corners=False,
        )[0].to(output.dtype)
        placement = torch.full(
            (channels, bottom - top, right - left),
            float("-inf"),
            device=output.device,
            dtype=output.dtype,
        )
        placement[:] = resized
        output[:, top:bottom, left:right] = torch.where(
            region.mask[top:bottom, left:right][None],
            placement,
            output[:, top:bottom, left:right],
        )
    return output


def seam_repair_labels(
    labels: torch.Tensor,
    footprint: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
    *,
    min_area: int = 8,
) -> torch.Tensor:
    """Remove seam fragments and close unassigned footprint pixels."""

    if footprint.ndim == 3:
        footprint = footprint[0]
    values = labels.detach().cpu().numpy().astype(np.int64)
    area = footprint.detach().cpu().numpy() > 0
    repaired = merge_room_fragments(values, taxonomy, min_area=min_area)
    room_ids = np.asarray(taxonomy.room_ids, dtype=np.int64)
    missing = (repaired == 0) & area
    rooms = np.isin(repaired, room_ids)
    if missing.any() and rooms.any():
        from scipy.ndimage import distance_transform_edt

        _, indices = distance_transform_edt(~rooms, return_indices=True)
        nearest = repaired[tuple(indices)]
        valid = missing & np.isin(nearest, room_ids)
        repaired[valid] = nearest[valid]
    return torch.from_numpy(repaired).to(labels.device, dtype=labels.dtype)


def decode_partitioned(
    footprint: torch.Tensor,
    regions: Sequence[PartitionRegion],
    decode_fn: Callable[[torch.Tensor], torch.Tensor],
    taxonomy: FloorplanTaxonomy | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Decode bounding-box crops through a callback and merge them."""

    if footprint.ndim == 3:
        footprint = footprint[0]
    local_labels = []
    for region in regions:
        top, left, bottom, right = region.bbox
        crop = footprint[top:bottom, left:right]
        crop_mask = region.mask[top:bottom, left:right]
        label = decode_fn(crop[None, None])
        if label.ndim == 4:
            label = label[0, 0]
        elif label.ndim == 3:
            label = label[0]
        label = F.interpolate(
            label[None, None].float(),
            size=crop.shape,
            mode="nearest",
        )[0, 0].to(footprint.device)
        label = torch.where(crop_mask, label, torch.zeros_like(label))
        local_labels.append(label)
    merged = merge_region_labels(footprint, regions, local_labels)
    if taxonomy is not None:
        merged = seam_repair_labels(merged, footprint, taxonomy)
    return merged, local_labels


__all__ = [
    "PartitionRegion",
    "decode_partitioned",
    "merge_region_labels",
    "merge_region_logits",
    "propose_batch_partitions",
    "propose_partition",
    "seam_repair_labels",
]
