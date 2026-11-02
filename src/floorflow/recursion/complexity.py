"""Learned direct/recursive/split decisions from floorplan complexity."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_erosion, label
from torch import nn

from floorflow.data.taxonomy import (
    FURNITURE_RARE,
    ROOM_RARE,
    FloorplanTaxonomy,
)

COMPLEXITY_FEATURE_NAMES = (
    "area_fraction",
    "boundary_fraction",
    "bbox_fill",
    "bbox_aspect_log",
    "perimeter_ratio",
    "corner_count_log",
    "convexity",
    "radial_dispersion",
    "room_class_count",
    "room_component_count",
    "room_entropy",
    "rare_density",
    "opening_density",
    "furniture_density",
)
POLICY_INPUT_FEATURE_NAMES = COMPLEXITY_FEATURE_NAMES[:8]

POLICY_ACTION_NAMES = ("direct", "recursive", "split")


def _component_count(mask: np.ndarray, min_area: int = 8) -> int:
    components, count = label(np.asarray(mask, dtype=bool))
    return sum(
        int((components == index).sum()) >= int(min_area)
        for index in range(1, int(count) + 1)
    )


def _entropy(counts: list[int]) -> float:
    total = float(sum(counts))
    if total <= 0:
        return 0.0
    values = [count / total for count in counts if count > 0]
    return float(-sum(value * math.log(value) for value in values))


def _footprint_features(mask: np.ndarray) -> list[float]:
    area = float(max(1, int(mask.sum())))
    coordinates = np.argwhere(mask)
    if coordinates.size:
        extent = coordinates.max(axis=0) - coordinates.min(axis=0) + 1
        bbox_area = float(max(1, int(extent[0] * extent[1])))
        aspect = float(max(extent) / max(1, min(extent)))
    else:
        bbox_area = 1.0
        aspect = 1.0
    boundary = mask & ~binary_erosion(mask, border_value=0)
    contours, _hierarchy = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    if contours:
        contour = max(contours, key=cv2.contourArea)
        perimeter = float(cv2.arcLength(contour, True))
        approximation = cv2.approxPolyDP(
            contour,
            max(0.002, 0.01 * perimeter),
            True,
        )
        corner_count = max(3, len(approximation))
        hull_area = float(max(1.0, cv2.contourArea(cv2.convexHull(contour))))
        convexity = area / hull_area
        centered = contour.reshape(-1, 2).astype(np.float64)
        centered -= centered.mean(axis=0, keepdims=True)
        radii = np.linalg.norm(centered, axis=1)
        radial_dispersion = float(
            radii.std() / max(1.0, radii.mean())
        )
    else:
        perimeter = 0.0
        corner_count = 3
        convexity = 1.0
        radial_dispersion = 0.0
    return [
        area / float(mask.size),
        float(boundary.sum()) / area,
        area / bbox_area,
        math.log(max(1.0, aspect)),
        perimeter / math.sqrt(area),
        math.log(float(max(3, corner_count))),
        min(1.0, max(0.0, convexity)),
        radial_dispersion,
    ]


def complexity_features(
    footprint: torch.Tensor,
    target: torch.Tensor | None = None,
    taxonomy: FloorplanTaxonomy | None = None,
) -> torch.Tensor:
    """Return per-case numeric complexity features for a batch."""

    if footprint.ndim == 4:
        footprint = footprint[:, 0]
    footprint_np = footprint.detach().cpu().numpy() > 0
    if footprint_np.ndim != 3:
        raise ValueError("footprint must be [B,H,W] or [B,1,H,W]")
    target_np = None
    if target is not None:
        if target.ndim == 4:
            target = target[:, 0]
        target_np = target.detach().cpu().numpy()
        if target_np.shape != footprint_np.shape:
            raise ValueError("target and footprint spatial shapes differ")
    if target is not None and taxonomy is None:
        raise ValueError("taxonomy is required when target labels are available")

    rows = []
    for index, mask in enumerate(footprint_np):
        features = _footprint_features(mask)
        if target_np is None or taxonomy is None:
            features.extend([0.0] * 6)
        else:
            labels = target_np[index]
            room_counts = [
                int((labels == class_id).sum())
                for class_id in taxonomy.room_ids
            ]
            present_rooms = [count for count in room_counts if count > 0]
            components = sum(
                _component_count(labels == class_id)
                for class_id in taxonomy.room_ids
            )
            rare = np.isin(
                labels,
                np.asarray(
                    [
                        taxonomy.class_id(ROOM_RARE),
                        taxonomy.class_id(FURNITURE_RARE),
                    ]
                ),
            )
            opening = np.isin(labels, np.asarray(taxonomy.opening_ids))
            furniture = np.isin(labels, np.asarray(taxonomy.furniture_ids))
            area = float(max(1, int(mask.sum())))
            features.extend(
                [
                    float(len(present_rooms)),
                    float(components),
                    _entropy(present_rooms),
                    float((rare & mask).sum()) / area,
                    float((opening & mask).sum()) / area,
                    float((furniture & mask).sum()) / area,
                ]
            )
        rows.append(features)
    return torch.tensor(rows, dtype=torch.float32, device=footprint.device)


def complexity_score(features: torch.Tensor) -> torch.Tensor:
    """Map raw features to a monotone, dimensionless complexity coordinate."""

    features = features.to(dtype=torch.float32)
    if features.ndim != 2 or features.shape[1] != len(COMPLEXITY_FEATURE_NAMES):
        raise ValueError("unexpected complexity feature shape")
    area = features[:, 0]
    boundary = features[:, 1]
    bbox_gap = (1.0 - features[:, 2]).clamp_min(0.0)
    aspect = features[:, 3]
    perimeter = features[:, 4].clamp_min(0.0)
    corners = features[:, 5].clamp_min(0.0)
    convex_gap = (1.0 - features[:, 6]).clamp_min(0.0)
    radial = features[:, 7].clamp_min(0.0)
    room_classes = torch.log1p(features[:, 8].clamp_min(0.0))
    room_components = torch.log1p(features[:, 9].clamp_min(0.0))
    room_entropy = features[:, 10].clamp_min(0.0)
    rare = features[:, 11].clamp_min(0.0)
    elements = (
        features[:, 12].clamp_min(0.0)
        + features[:, 13].clamp_min(0.0)
    )
    return (
        area
        + 2.0 * boundary
        + bbox_gap
        + 0.35 * aspect
        + 0.15 * perimeter
        + 0.3 * corners
        + convex_gap
        + radial
        + 0.8 * room_classes
        + 0.6 * room_components
        + 0.4 * room_entropy
        + rare
        + torch.log1p(elements)
    )


@dataclass(frozen=True)
class PolicyLabelCalibration:
    low_quantile: float
    high_quantile: float
    low_threshold: float
    high_threshold: float

    @classmethod
    def fit(
        cls,
        features: torch.Tensor,
        *,
        low_quantile: float = 1.0 / 3.0,
        high_quantile: float = 2.0 / 3.0,
    ) -> PolicyLabelCalibration:
        scores = complexity_score(features).detach().cpu().numpy()
        return cls(
            low_quantile=float(low_quantile),
            high_quantile=float(high_quantile),
            low_threshold=float(np.quantile(scores, low_quantile)),
            high_threshold=float(np.quantile(scores, high_quantile)),
        )


def derive_policy_labels(
    features: torch.Tensor,
    calibration: PolicyLabelCalibration,
) -> torch.Tensor:
    """Derive train-only direct/recursive/split labels from score quantiles."""

    scores = complexity_score(features)
    labels = torch.ones(scores.shape[0], dtype=torch.long, device=scores.device)
    labels = torch.where(
        scores <= float(calibration.low_threshold),
        torch.zeros_like(labels),
        labels,
    )
    labels = torch.where(
        scores > float(calibration.high_threshold),
        torch.full_like(labels, 2),
        labels,
    )
    return labels


class RecursionPolicy(nn.Module):
    """Small learned policy over direct, recursive and split actions."""

    def __init__(
        self,
        feature_size: int = len(POLICY_INPUT_FEATURE_NAMES),
        hidden_size: int = 32,
        num_actions: int = 3,
    ) -> None:
        super().__init__()
        self.feature_size = int(feature_size)
        self.num_actions = int(num_actions)
        self.register_buffer("feature_mean", torch.zeros(self.feature_size))
        self.register_buffer("feature_scale", torch.ones(self.feature_size))
        self.network = nn.Sequential(
            nn.Linear(self.feature_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, self.num_actions),
        )

    def set_standardizer(
        self,
        mean: torch.Tensor,
        scale: torch.Tensor,
    ) -> None:
        if mean.shape != (self.feature_size,) or scale.shape != (
            self.feature_size,
        ):
            raise ValueError("standardizer shape does not match feature size")
        self.feature_mean.copy_(mean.detach())
        self.feature_scale.copy_(scale.clamp_min(1.0e-6).detach())

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        standardized = (
            features.float() - self.feature_mean
        ) / self.feature_scale
        return self.network(standardized)

    @torch.no_grad()
    def decide(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = self(features)
        probability = logits.softmax(dim=-1)
        action = probability.argmax(dim=-1)
        normalized = action.float() / float(max(1, self.num_actions - 1))
        return {
            "logits": logits,
            "probability": probability,
            "action": action,
            "recursion_strength": normalized,
        }


def policy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if logits.shape[0] != labels.shape[0]:
        raise ValueError("policy logits and labels batch sizes differ")
    return F.cross_entropy(logits, labels.long(), weight=class_weights)


__all__ = [
    "COMPLEXITY_FEATURE_NAMES",
    "POLICY_ACTION_NAMES",
    "POLICY_INPUT_FEATURE_NAMES",
    "PolicyLabelCalibration",
    "RecursionPolicy",
    "complexity_features",
    "complexity_score",
    "derive_policy_labels",
    "policy_loss",
]
