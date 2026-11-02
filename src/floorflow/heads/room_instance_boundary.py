"""Learned room-instance boundaries over an existing room segmenter."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import label
from torch import nn

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.heads.room_segmenter import (
    ContourRoomSegmenter,
    ConvNormAct,
    RoomEncoderFeatures,
    prepare_room_condition,
)


class RoomInstanceBoundaryHead(nn.Module):
    """Predict thin boundaries between adjacent room instances."""

    def __init__(self, channels: int, hidden_channels: int = 32) -> None:
        super().__init__()
        hidden = int(hidden_channels)
        self.network = nn.Sequential(
            ConvNormAct(int(channels), hidden),
            ConvNormAct(hidden, hidden),
            nn.Conv2d(hidden, 1, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class MultiScaleRoomBoundaryHead(nn.Module):
    """Decode thin boundaries from full-resolution and encoder skip features."""

    def __init__(
        self,
        decoder_channels: int,
        *,
        hidden_channels: int = 32,
    ) -> None:
        super().__init__()
        hidden = int(hidden_channels)
        self.hidden_projection = nn.Conv2d(int(decoder_channels), hidden, 1)
        self.x1_projection = nn.Conv2d(64, hidden, 1)
        self.x2_projection = nn.Conv2d(128, hidden, 1)
        self.x3_projection = nn.Conv2d(256, hidden, 1)
        self.fusion = nn.Sequential(
            ConvNormAct(hidden * 4, hidden * 2),
            ConvNormAct(hidden * 2, hidden),
            nn.Conv2d(hidden, 1, 1),
        )

    def forward(self, features: RoomEncoderFeatures) -> torch.Tensor:
        size = features.hidden.shape[-2:]
        levels = [
            features.hidden,
            self.x1_projection(features.x1),
            self.x2_projection(features.x2),
            self.x3_projection(features.x3),
        ]
        levels = [
            level
            if level.shape[-2:] == size
            else F.interpolate(
                level,
                size=size,
                mode="bilinear",
                align_corners=False,
            )
            for level in levels
        ]
        levels[0] = self.hidden_projection(levels[0])
        return self.fusion(torch.cat(levels, dim=1))


class ImageBoundaryDecoder(nn.Module):
    """Small independent boundary decoder over the native drawing."""

    def __init__(
        self,
        *,
        input_channels: int = 3,
        hidden_channels: int = 16,
    ) -> None:
        super().__init__()
        hidden = int(hidden_channels)
        self.network = nn.Sequential(
            ConvNormAct(int(input_channels), hidden),
            ConvNormAct(hidden, hidden),
            ConvNormAct(hidden, hidden),
            nn.Conv2d(hidden, 1, 1),
        )

    def forward(self, control_image: torch.Tensor) -> torch.Tensor:
        return self.network(control_image)


class BoundaryAwareRoomSegmenter(nn.Module):
    """Frozen dense room segmenter plus a trainable boundary head."""

    def __init__(
        self,
        taxonomy: FloorplanTaxonomy,
        *,
        decoder_channels: int = 128,
        boundary_channels: int = 32,
        boundary_mode: str = "final",
    ) -> None:
        super().__init__()
        self.room = ContourRoomSegmenter(
            len(taxonomy.room_ids),
            pretrained=False,
            decoder_channels=int(decoder_channels),
        )
        if boundary_mode not in {
            "final",
            "multiscale",
            "independent",
            "dense",
        }:
            raise ValueError(
                "boundary_mode must be final, multiscale, independent, or dense"
            )
        self.boundary_mode = boundary_mode
        if boundary_mode == "multiscale":
            boundary = MultiScaleRoomBoundaryHead(
                int(decoder_channels),
                hidden_channels=int(boundary_channels),
            )
        elif boundary_mode == "independent":
            boundary = ImageBoundaryDecoder(
                hidden_channels=int(boundary_channels),
            )
        elif boundary_mode == "dense":
            boundary = ImageBoundaryDecoder(
                input_channels=3 + len(taxonomy.room_ids) + 1,
                hidden_channels=int(boundary_channels),
            )
        else:
            boundary = RoomInstanceBoundaryHead(
                int(decoder_channels),
                hidden_channels=int(boundary_channels),
            )
        self.boundary = boundary

    def forward(
        self,
        footprint: torch.Tensor,
        control_image: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        features = self.room.encode_control_features(
            footprint,
            control_image=control_image,
        )
        dense_logits = self.room.final(features.hidden)
        if self.boundary_mode in {"independent", "dense"}:
            boundary_input = (
                prepare_room_condition(footprint)
                if control_image is None
                else control_image.float()
            )
            if self.boundary_mode == "dense":
                footprint_4d = (
                    footprint[:, None].float()
                    if footprint.ndim == 3
                    else footprint.float()
                )
                boundary_input = torch.cat(
                    [
                        boundary_input,
                        dense_logits.softmax(dim=1),
                        footprint_4d,
                    ],
                    dim=1,
                )
            boundary_logits = self.boundary(boundary_input)
        elif self.boundary_mode == "multiscale":
            boundary_logits = self.boundary(features)
        else:
            boundary_logits = self.boundary(features.hidden)
        return {
            "dense_logits": dense_logits,
            "boundary_logits": boundary_logits,
        }


def room_instance_boundary_target(
    labels: torch.Tensor,
    footprint: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
    *,
    radius: int = 2,
) -> torch.Tensor:
    """Return room-side seams separating same-class room instances."""

    label_rows = labels.detach().cpu().numpy()
    footprint_rows = footprint.detach().cpu().numpy().astype(bool)
    output = []
    for row, valid in zip(label_rows, footprint_rows):
        instances = np.zeros(row.shape, dtype=np.int32)
        instance_classes: dict[int, int] = {}
        next_id = 1
        for class_id in taxonomy.room_ids:
            components, count = label(row == int(class_id))
            present = components > 0
            instances[present] = components[present] + next_id - 1
            for component_id in range(1, int(count) + 1):
                instance_classes[next_id + component_id - 1] = int(class_id)
            next_id += int(count)
        room = instances > 0
        boundary = np.zeros(row.shape, dtype=bool)
        height, width = row.shape
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            y0, y1 = max(0, -dy), min(height, height - dy)
            x0, x1 = max(0, -dx), min(width, width - dx)
            current = instances[y0:y1, x0:x1]
            neighbor = instances[y0 + dy : y1 + dy, x0 + dx : x1 + dx]
            neighbor_valid = valid[y0 + dy : y1 + dy, x0 + dx : x1 + dx]
            current_valid = room[y0:y1, x0:x1] & valid[y0:y1, x0:x1]
            boundary[y0:y1, x0:x1] |= current_valid & (
                (current != neighbor) | ~neighbor_valid
            )
        radius = max(1, int(radius))
        filtered = np.zeros(row.shape, dtype=bool)
        for y, x in np.argwhere(boundary):
            own_instance = int(instances[y, x])
            if own_instance <= 0:
                continue
            own_class = instance_classes[own_instance]
            neighborhood = instances[
                max(0, y - radius) : y + radius + 1,
                max(0, x - radius) : x + radius + 1,
            ]
            other = {
                int(value)
                for value in np.unique(neighborhood)
                if int(value) > 0
                and int(value) != own_instance
                and instance_classes[int(value)] == own_class
            }
            filtered[y, x] = bool(other)
        output.append(filtered)
    return torch.from_numpy(np.stack(output)).to(labels.device)


def room_boundary_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    footprint: torch.Tensor,
    *,
    positive_weight: float = 4.0,
    tversky_weight: float = 0.0,
    tversky_alpha: float = 0.3,
    tversky_beta: float = 0.7,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Balanced BCE and Dice for thin room-instance boundaries."""

    if logits.ndim == 4:
        logits = logits[:, 0]
    if footprint.ndim == 3:
        footprint = footprint.bool()
    target = target.bool() & footprint.bool()
    bce = F.binary_cross_entropy_with_logits(
        logits,
        target.float(),
        pos_weight=logits.new_tensor(float(positive_weight)),
    )
    probability = logits.sigmoid() * footprint.float()
    expected = target.float()
    intersection = (probability * expected).sum(dim=(1, 2))
    denominator = probability.sum(dim=(1, 2)) + expected.sum(dim=(1, 2))
    dice = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
    true_positive = (probability * expected).sum(dim=(1, 2))
    false_positive = (probability * (1.0 - expected)).sum(dim=(1, 2))
    false_negative = ((1.0 - probability) * expected).sum(dim=(1, 2))
    tversky = 1.0 - (
        (true_positive + 1.0)
        / (
            true_positive
            + float(tversky_alpha) * false_positive
            + float(tversky_beta) * false_negative
            + 1.0
        )
    ).mean()
    total = bce + dice.mean() + float(tversky_weight) * tversky
    return total, {
        "room_boundary_loss": total.detach(),
        "room_boundary_bce": bce.detach(),
        "room_boundary_dice": dice.mean().detach(),
        "room_boundary_tversky": tversky.detach(),
        "room_boundary_positive_rate": expected.mean().detach(),
    }


def split_room_instances(
    labels: torch.Tensor,
    boundary_logits: torch.Tensor,
    footprint: torch.Tensor,
    room_ids: torch.Tensor,
    *,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Clear predicted boundary pixels while preserving dense room classes."""

    if boundary_logits.ndim == 4:
        boundary_logits = boundary_logits[:, 0]
    if footprint.ndim != 3:
        footprint = footprint[:, 0]
    room_support = torch.isin(labels, room_ids) & footprint.bool()
    boundary = (boundary_logits.sigmoid() >= float(threshold)) & room_support
    return torch.where(boundary, torch.zeros_like(labels), labels)


def room_boundary_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    footprint: torch.Tensor,
    *,
    threshold: float,
) -> dict[str, float]:
    """Return boundary precision, recall, F1, and predicted-positive rate."""

    if logits.ndim == 4:
        logits = logits[:, 0]
    valid = footprint.bool() if footprint.ndim == 3 else footprint[:, 0].bool()
    predicted = (logits.sigmoid() >= float(threshold)) & valid
    expected = target.bool() & valid
    true_positive = (predicted & expected).sum()
    false_positive = (predicted & ~expected).sum()
    false_negative = (~predicted & expected).sum()
    precision = true_positive.float() / (true_positive + false_positive).clamp_min(1)
    recall = true_positive.float() / (true_positive + false_negative).clamp_min(1)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1.0e-8)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "positive_rate": float(predicted.float().mean()),
    }


__all__ = [
    "BoundaryAwareRoomSegmenter",
    "ImageBoundaryDecoder",
    "MultiScaleRoomBoundaryHead",
    "RoomInstanceBoundaryHead",
    "room_boundary_loss",
    "room_boundary_metrics",
    "room_instance_boundary_target",
    "split_room_instances",
]
