"""Dense room-partition targets, losses, and semantic composition."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

from floorflow.data.taxonomy import FloorplanTaxonomy


class RoomPartitionHead(nn.Module):
    """Predict room type at every interior pixel independently of objects."""

    def __init__(
        self,
        in_channels: int,
        num_room_classes: int,
        prior_channels: int = 0,
    ) -> None:
        super().__init__()
        self.prior_channels = int(prior_channels)
        if self.prior_channels:
            self.prior_projection = nn.Conv2d(self.prior_channels, in_channels, 1)
            nn.init.zeros_(self.prior_projection.weight)
            nn.init.zeros_(self.prior_projection.bias)
            self.prior_logits = nn.Conv2d(self.prior_channels, num_room_classes, 1, bias=False)
            nn.init.zeros_(self.prior_logits.weight)
            with torch.no_grad():
                for index in range(min(self.prior_channels, num_room_classes)):
                    self.prior_logits.weight[index, index, 0, 0] = 2.0
        hidden = max(32, int(in_channels) // 2)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.GroupNorm(max(1, hidden // 16), hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GroupNorm(max(1, hidden // 16), hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, num_room_classes, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features: torch.Tensor, prior: torch.Tensor | None = None) -> torch.Tensor:
        if self.prior_channels:
            if prior is None:
                raise ValueError("room prior is required by this partition head")
            if prior.shape[1] != self.prior_channels:
                raise ValueError(f"expected {self.prior_channels} prior channels, got {prior.shape[1]}")
            features = features + self.prior_projection(prior.to(features.dtype))
        output = self.net(features)
        if self.prior_channels:
            output = output + self.prior_logits(prior.to(features.dtype))
        return output


def build_room_partition_target(
    labels: np.ndarray,
    footprint: np.ndarray,
    taxonomy: FloorplanTaxonomy,
    *,
    fill_occluded: bool = True,
) -> np.ndarray:
    """Lift visible room labels to a dense partition with nearest-room completion."""

    labels = np.asarray(labels)
    footprint = np.asarray(footprint, dtype=bool)
    target = np.full(labels.shape, -100, dtype=np.int64)
    for room_index, class_id in enumerate(taxonomy.room_ids):
        target[labels == class_id] = room_index
    eligible = footprint & (target < 0)
    known = target >= 0
    if fill_occluded and eligible.any() and known.any():
        indices = distance_transform_edt(~known, return_distances=False, return_indices=True)
        target[eligible] = target[tuple(indices[:, eligible])]
    target[~footprint] = -100
    return target


def batch_room_partition_target(
    labels: torch.Tensor,
    footprint: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
    *,
    fill_occluded: bool = True,
) -> torch.Tensor:
    rows = [
        build_room_partition_target(
            item,
            area,
            taxonomy,
            fill_occluded=fill_occluded,
        )
        for item, area in zip(labels.detach().cpu().numpy(), footprint.detach().cpu().numpy())
    ]
    return torch.from_numpy(np.stack(rows)).to(labels.device)


def room_partition_confidence(
    labels: torch.Tensor,
    footprint: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
    *,
    visible_weight: float = 1.0,
    furniture_weight: float = 0.15,
    wall_weight: float = 0.05,
    opening_weight: float = 0.05,
    other_weight: float = 0.10,
) -> torch.Tensor:
    """Weight observed room labels above nearest-room pseudo labels."""

    room_ids = torch.tensor(taxonomy.room_ids, device=labels.device)
    furniture_ids = torch.tensor(taxonomy.furniture_ids, device=labels.device)
    opening_ids = torch.tensor(taxonomy.opening_ids, device=labels.device)
    wall_id = taxonomy.class_id("wall")
    weights = torch.full_like(labels, float(other_weight), dtype=torch.float32)
    weights[torch.isin(labels, room_ids)] = float(visible_weight)
    weights[torch.isin(labels, furniture_ids)] = float(furniture_weight)
    weights[torch.isin(labels, opening_ids)] = float(opening_weight)
    weights[labels == wall_id] = float(wall_weight)
    weights[~(footprint > 0.5)] = 0.0
    return weights


def room_partition_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    class_weights: torch.Tensor | None = None,
    dice_weight: float = 1.0,
    tversky_weight: float = 0.0,
    pixel_weights: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Case-macro cross-entropy, Dice, and Tversky room losses."""

    valid = target >= 0
    if not valid.any():
        zero = logits.sum() * 0.0
        return {
            "loss": zero,
            "partition_ce": zero.detach(),
            "partition_dice": zero.detach(),
            "partition_tversky": zero.detach(),
        }
    ce_pixels = F.cross_entropy(
        logits,
        target.clamp_min(0),
        weight=class_weights,
        reduction="none",
    )
    if pixel_weights is None:
        weights = valid.float()
    else:
        weights = pixel_weights.to(device=logits.device, dtype=logits.dtype) * valid.float()
    ce_by_case = (ce_pixels * weights).flatten(1).sum(dim=1) / (
        weights.flatten(1).sum(dim=1).clamp_min(1.0)
    )
    ce = ce_by_case.mean()
    probabilities = logits.softmax(dim=1)
    one_hot = F.one_hot(target.clamp_min(0), logits.shape[1]).permute(0, 3, 1, 2).float()
    weighted_valid = weights[:, None]
    one_hot = one_hot * weighted_valid
    intersection = (probabilities * one_hot).sum(dim=(2, 3))
    denominator = (
        (probabilities * weighted_valid).sum(dim=(2, 3))
        + one_hot.sum(dim=(2, 3))
    )
    present = one_hot.sum(dim=(2, 3)) > 0
    dice = (2.0 * intersection + 1.0) / (denominator + 1.0)
    dice_per_case = torch.where(
        present,
        1.0 - dice,
        torch.zeros_like(dice),
    )
    dice_by_case = dice_per_case.sum(dim=1) / present.sum(dim=1).clamp_min(1)
    dice_loss = dice_by_case.mean()
    true_positive = intersection
    expected_mass = one_hot.sum(dim=(2, 3))
    predicted_mass = (probabilities * weighted_valid).sum(dim=(2, 3))
    false_positive = (predicted_mass - true_positive).clamp_min(0.0)
    false_negative = (expected_mass - true_positive).clamp_min(0.0)
    tversky = (true_positive + 1.0) / (
        true_positive
        + 0.3 * false_positive
        + 0.7 * false_negative
        + 1.0
    )
    tversky_per_case = torch.where(
        present,
        (1.0 - tversky).pow(1.5),
        torch.zeros_like(tversky),
    )
    tversky_by_case = (
        tversky_per_case.sum(dim=1) / present.sum(dim=1).clamp_min(1)
    )
    tversky_loss = tversky_by_case.mean()
    total = (
        ce
        + float(dice_weight) * dice_loss
        + float(tversky_weight) * tversky_loss
    )
    return {
        "loss": total,
        "partition_ce": ce.detach(),
        "partition_dice": dice_loss.detach(),
        "partition_tversky": tversky_loss.detach(),
        "partition_ce_by_case": ce_by_case.detach(),
        "partition_dice_by_case": dice_by_case.detach(),
        "partition_tversky_by_case": tversky_by_case.detach(),
    }


@torch.no_grad()
def room_partition_metrics(
    predictions: torch.Tensor,
    target: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
) -> dict[str, float]:
    valid = target >= 0
    intersections = []
    unions = []
    correct = 0
    count = 0
    predicted = predictions.argmax(dim=1)
    for class_index in range(len(taxonomy.room_ids)):
        predicted_mask = (predicted == class_index) & valid
        target_mask = target == class_index
        if not target_mask.any():
            continue
        intersections.append(float((predicted_mask & target_mask).sum().cpu()))
        unions.append(float((predicted_mask | target_mask).sum().cpu()))
    correct = int(((predicted == target) & valid).sum().cpu())
    count = int(valid.sum().cpu())
    ious = [left / max(1.0, right) for left, right in zip(intersections, unions)]
    return {
        "room_partition_iou": float(np.mean(ious)) if ious else 0.0,
        "room_partition_accuracy": correct / max(1, count),
    }


def compose_partition_with_semantics(
    partition_logits: torch.Tensor,
    semantic_labels: torch.Tensor,
    footprint: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
) -> torch.Tensor:
    """Use partition for room fill and semantic heads for object overlays."""

    room_ids = torch.tensor(taxonomy.room_ids, device=partition_logits.device)
    partition_labels = room_ids[partition_logits.argmax(dim=1)]
    non_room = ~torch.isin(semantic_labels, room_ids) & (semantic_labels != 0)
    composed = torch.where(non_room, semantic_labels, partition_labels)
    return composed * (footprint > 0.5).long()
