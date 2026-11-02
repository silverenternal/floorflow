"""Explicit corner and segment encoding for exterior floorplan contours."""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn


def contour_geometry_tokens(
    footprints: torch.Tensor,
    *,
    max_points: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract ordered normalized corners and validity masks from raster contours."""

    values = footprints.detach().cpu().numpy()
    if values.ndim == 4:
        values = values[:, 0]
    batch, height, width = values.shape
    points = np.zeros((batch, max_points, 2), dtype=np.float32)
    valid = np.zeros((batch, max_points), dtype=bool)
    for batch_index in range(batch):
        mask = (values[batch_index] > 0.5).astype(np.uint8)
        contours, _hierarchy = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        perimeter = max(1.0, cv2.arcLength(contour, True))
        approximation = cv2.approxPolyDP(
            contour,
            max(0.002, 0.01 * perimeter),
            True,
        ).reshape(-1, 2)
        if len(approximation) < 3:
            approximation = contour.reshape(-1, 2)
        if len(approximation) > max_points:
            indices = np.linspace(
                0,
                len(approximation),
                max_points,
                endpoint=False,
                dtype=np.int64,
            )
            approximation = approximation[indices]
        count = len(approximation)
        points[batch_index, :count, 0] = approximation[:, 0] / max(1.0, width - 1.0)
        points[batch_index, :count, 1] = approximation[:, 1] / max(1.0, height - 1.0)
        valid[batch_index, :count] = True
    return (
        torch.from_numpy(points).to(footprints.device),
        torch.from_numpy(valid).to(footprints.device),
    )


def contour_geometric_features(
    points: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Build translation-normalized corner, edge, and turning-angle features."""

    previous = torch.roll(points, 1, dims=1)
    following = torch.roll(points, -1, dims=1)
    previous_edge = points - previous
    following_edge = following - points
    previous_length = previous_edge.norm(dim=-1, keepdim=True)
    following_length = following_edge.norm(dim=-1, keepdim=True)
    previous_unit = previous_edge / previous_length.clamp_min(1.0e-6)
    following_unit = following_edge / following_length.clamp_min(1.0e-6)
    cosine = (previous_unit * following_unit).sum(dim=-1, keepdim=True)
    sine = previous_unit[..., 0:1] * following_unit[..., 1:2] - (
        previous_unit[..., 1:2] * following_unit[..., 0:1]
    )
    center = (points * valid[..., None]).sum(dim=1, keepdim=True) / (
        valid.sum(dim=1, keepdim=True).clamp_min(1)[..., None]
    )
    centered = points - center
    features = torch.cat(
        [
            centered,
            points,
            previous_unit,
            following_unit,
            previous_length.clamp_max(1.0),
            following_length.clamp_max(1.0),
            cosine,
            sine,
            valid[..., None].to(points.dtype),
        ],
        dim=-1,
    )
    return features.masked_fill(~valid[..., None], 0.0)


class BoundaryGeometryEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 96,
        max_points: int = 64,
    ) -> None:
        super().__init__()
        self.max_points = int(max_points)
        self.hidden_dim = int(hidden_dim)
        self.position = nn.Parameter(
            torch.randn(1, self.max_points, hidden_dim) * 0.02
        )
        self.network = nn.Sequential(
            nn.Linear(13, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, footprints: torch.Tensor) -> dict[str, torch.Tensor]:
        points, valid = contour_geometry_tokens(
            footprints,
            max_points=self.max_points,
        )
        features = contour_geometric_features(points, valid)
        tokens = self.network(features) + self.position.to(features.dtype)
        return {
            "points": points,
            "valid": valid,
            "tokens": tokens,
        }
