"""Build query-level element masks and types from CAD rasters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.ndimage import label

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.distribution.hierarchy import ChannelHierarchy


@dataclass(frozen=True)
class ElementTargets:
    masks: torch.Tensor
    types: torch.Tensor
    valid: torch.Tensor


def _group_targets(
    labels: np.ndarray,
    class_ids: tuple[int, ...],
    max_queries: int,
    min_area: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = labels.shape
    rows = []
    for type_index, class_id in enumerate(class_ids):
        components, count = label(labels == class_id)
        for component_index in range(1, int(count) + 1):
            mask = components == component_index
            area = int(mask.sum())
            if area < min_area:
                continue
            rows.append((area, type_index, mask))
    rows.sort(key=lambda row: row[0], reverse=True)
    rows = rows[: int(max_queries)]
    masks = np.zeros((max_queries, height, width), dtype=np.float32)
    types = np.zeros(max_queries, dtype=np.int64)
    valid = np.zeros(max_queries, dtype=bool)
    for index, (_area, type_index, mask) in enumerate(rows):
        masks[index] = mask
        types[index] = type_index
        valid[index] = True
    return masks, types, valid


def element_targets(
    labels: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
    hierarchy: ChannelHierarchy,
    *,
    max_queries: dict[str, int] | None = None,
    min_area: int = 8,
) -> dict[str, ElementTargets]:
    counts = max_queries or {}
    group_ids = {
        "wall": hierarchy.wall_ids,
        "opening": hierarchy.opening_ids,
        "room": hierarchy.room_ids,
        "furniture": hierarchy.furniture_ids,
        "support": hierarchy.support_ids,
    }
    output = {}
    for name in ("wall", "opening", "room", "furniture", "support"):
        query_count = int(counts.get(name, 8))
        batch_rows = [
            _group_targets(
                item.detach().cpu().numpy(),
                group_ids[name],
                query_count,
                min_area,
            )
            for item in labels
        ]
        output[name] = ElementTargets(
            masks=torch.from_numpy(np.stack([row[0] for row in batch_rows])).to(
                labels.device
            ),
            types=torch.from_numpy(np.stack([row[1] for row in batch_rows])).to(
                labels.device
            ),
            valid=torch.from_numpy(np.stack([row[2] for row in batch_rows])).to(
                labels.device
            ),
        )
    return output


__all__ = ["ElementTargets", "element_targets"]
