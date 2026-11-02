"""Data-only identifiability diagnostics for contour-conditioned layouts."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from floorflow.data.taxonomy import FloorplanTaxonomy


@dataclass(frozen=True)
class ContourRecord:
    name: str
    exact_hash: int
    descriptor: torch.Tensor
    room_labels: torch.Tensor
    room_program: torch.Tensor


def contour_descriptor(footprint: torch.Tensor, size: int = 32) -> torch.Tensor:
    if footprint.ndim == 2:
        footprint = footprint[None]
    if footprint.ndim == 3:
        footprint = footprint[None]
    if footprint.ndim != 4 or footprint.shape[1] != 1:
        raise ValueError("footprint must be [H,W], [1,H,W], or [1,1,H,W]")
    occupancy = F.adaptive_avg_pool2d(
        footprint.float(),
        (int(size), int(size)),
    )
    area = footprint.float().mean().reshape(1)
    return torch.cat([occupancy.flatten(), area])


def room_program(
    labels: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
) -> torch.Tensor:
    room_ids = torch.tensor(taxonomy.room_ids, dtype=labels.dtype)
    return torch.isin(labels, room_ids).float()


def room_labels_at_size(
    labels: torch.Tensor,
    size: int = 64,
) -> torch.Tensor:
    return F.interpolate(
        labels[None, None].float(),
        size=(int(size), int(size)),
        mode="nearest",
    )[0, 0].long()


def macro_iou(
    prediction: torch.Tensor,
    target: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
) -> float:
    values = []
    for class_id in taxonomy.room_ids:
        expected = target == int(class_id)
        if not bool(expected.any()):
            continue
        predicted = prediction == int(class_id)
        intersection = (predicted & expected).sum().item()
        union = (predicted | expected).sum().item()
        values.append(intersection / max(1, union))
    return float(np.mean(values)) if values else 0.0


def entropy(values: list[object]) -> float:
    if not values:
        return 0.0
    counts = Counter(values)
    total = float(sum(counts.values()))
    probabilities = np.asarray(
        [count / total for count in counts.values()],
        dtype=np.float64,
    )
    return float(-(probabilities * np.log(probabilities)).sum())


def exact_group_entropy(
    records: list[ContourRecord],
    *,
    group_field: str = "room_program",
) -> dict[str, float]:
    groups: dict[int, list[ContourRecord]] = defaultdict(list)
    for record in records:
        groups[record.exact_hash].append(record)
    weighted = []
    weights = []
    for members in groups.values():
        if len(members) < 2:
            continue
        if group_field == "room_program":
            values = [
                tuple(int(value) for value in member.room_program.tolist())
                for member in members
            ]
        else:
            values = [member.room_labels for member in members]
        weighted.append(entropy(values))
        weights.append(len(members))
    if not weighted:
        return {"groups": 0.0, "weighted_entropy": 0.0}
    return {
        "groups": float(len(weighted)),
        "weighted_entropy": float(np.average(weighted, weights=weights)),
    }


def duplicate_contour_layout_ious(
    records: list[ContourRecord],
    taxonomy: FloorplanTaxonomy,
) -> dict[str, float | int]:
    """Measure target disagreement among exactly identical contours."""

    groups: dict[int, list[ContourRecord]] = defaultdict(list)
    for record in records:
        groups[record.exact_hash].append(record)
    values = []
    ambiguous_pairs = 0
    for members in groups.values():
        for left in range(len(members)):
            for right in range(left + 1, len(members)):
                iou = macro_iou(
                    members[left].room_labels,
                    members[right].room_labels,
                    taxonomy,
                )
                values.append(iou)
                ambiguous_pairs += int(iou < 0.95)
    if not values:
        return {
            "duplicate_pairs": 0,
            "fraction_below_095": 0.0,
            "mean_iou": 1.0,
            "minimum_iou": 1.0,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "duplicate_pairs": int(array.size),
        "fraction_below_095": float(ambiguous_pairs / array.size),
        "mean_iou": float(array.mean()),
        "minimum_iou": float(array.min()),
        "p10_iou": float(np.quantile(array, 0.10)),
    }


def nearest_indices(
    descriptors: torch.Tensor,
    bank: torch.Tensor,
    *,
    k: int = 1,
    chunk_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = []
    indices = []
    for start in range(0, descriptors.shape[0], int(chunk_size)):
        distance = torch.cdist(
            descriptors[start : start + chunk_size].float(),
            bank.float(),
        )
        selected = distance.topk(min(int(k), bank.shape[0]), dim=1, largest=False)
        values.append(selected.values.cpu())
        indices.append(selected.indices.cpu())
    return torch.cat(indices), torch.cat(values)


def program_neighbor_entropy(
    test_program: torch.Tensor,
    train_program: torch.Tensor,
    neighbors: torch.Tensor,
) -> np.ndarray:
    probabilities = train_program[neighbors.reshape(-1)].reshape(
        neighbors.shape[0],
        neighbors.shape[1],
        -1,
    ).mean(dim=1)
    p = probabilities.double().clamp(1.0e-8, 1.0 - 1.0e-8)
    return (
        -(p * p.log() + (1.0 - p) * (1.0 - p).log())
        / np.log(2.0)
    ).mean(dim=1).numpy()


__all__ = [
    "ContourRecord",
    "contour_descriptor",
    "duplicate_contour_layout_ious",
    "entropy",
    "exact_group_entropy",
    "macro_iou",
    "nearest_indices",
    "program_neighbor_entropy",
    "room_labels_at_size",
    "room_program",
]
