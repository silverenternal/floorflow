"""Detection-style heatmaps for junctions, wall centerlines, and openings."""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_erosion
from scipy.ndimage import label as component_label
from skan import Skeleton
from skimage.morphology import skeletonize
from torch import nn

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.heads.topology_moe import SpatialExpertMoE

HEATMAP_KEYS = (
    "junction",
    "wall_centerline",
    "opening_center",
    "room_boundary",
)


class GeometryHeatmapHeads(nn.Module):
    """Shared geometry trunk with four dense detection heads."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 64,
        experts: int = 1,
        top_k: int = 1,
    ) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.GroupNorm(max(1, hidden_dim // 16), hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(max(1, hidden_dim // 16), hidden_dim),
            nn.SiLU(),
        )
        self.heatmaps = nn.ModuleDict(
            {
                name: nn.Conv2d(hidden_dim, 1, 1)
                for name in HEATMAP_KEYS
            }
        )
        self.junction_offsets = nn.Conv2d(hidden_dim, 2, 1)
        self.moe = (
            SpatialExpertMoE(hidden_dim, 2, experts=experts, top_k=top_k)
            if experts > 1
            else None
        )
        for head in (*self.heatmaps.values(), self.junction_offsets):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.trunk(features)
        predictions = {
            **{
                f"{name}_logits": head(hidden)
                for name, head in self.heatmaps.items()
            },
            "junction_offsets": self.junction_offsets(hidden),
        }
        if self.moe is not None:
            residual, diagnostics = self.moe(hidden)
            predictions["wall_centerline_logits"] = (
                predictions["wall_centerline_logits"] + residual[:, :1]
            )
            predictions["opening_center_logits"] = (
                predictions["opening_center_logits"] + residual[:, 1:]
            )
            predictions["moe_diagnostics"] = diagnostics
        return predictions


def _gaussian_heatmap(mask: np.ndarray, sigma: float) -> np.ndarray:
    values = np.asarray(mask, dtype=np.float32)
    if not values.any():
        return values
    heatmap = cv2.GaussianBlur(
        values,
        (0, 0),
        sigmaX=max(0.5, float(sigma)),
        sigmaY=max(0.5, float(sigma)),
    ).clip(0.0, 1.0)
    return heatmap / max(float(heatmap.max()), 1.0e-6)


def _junction_mask(skeleton: np.ndarray) -> np.ndarray:
    if not skeleton.any():
        return np.zeros_like(skeleton, dtype=np.uint8)
    graph = Skeleton(skeleton.astype(np.uint8))
    mask = np.zeros_like(skeleton, dtype=np.uint8)
    for coordinate, degree in zip(graph.coordinates, graph.degrees):
        if int(degree) != 2:
            y, x = map(int, coordinate)
            mask[y, x] = 1
    return mask


def wall_junction_targets(
    wall: torch.Tensor,
    footprint: torch.Tensor,
    *,
    sigma: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build junction and degree-class targets from a wall raster."""

    if wall.ndim == 4:
        wall = wall[:, 0]
    if footprint.ndim == 4:
        footprint = footprint[:, 0]
    wall_array = wall.detach().cpu().numpy() > 0.5
    footprint_array = footprint.detach().cpu().bool().numpy()
    heatmaps = []
    degree_maps = []
    for drawing, area in zip(wall_array, footprint_array, strict=True):
        centerline = skeletonize(drawing & area)
        degree_map = np.zeros_like(centerline, dtype=np.int64)
        junction = np.zeros_like(centerline, dtype=np.uint8)
        if centerline.any():
            graph = Skeleton(centerline.astype(np.uint8))
            for coordinate, degree in zip(
                graph.coordinates,
                graph.degrees,
                strict=True,
            ):
                y, x = map(int, coordinate)
                degree_map[y, x] = min(int(degree), 4)
                if int(degree) != 2:
                    junction[y, x] = 1
        heatmaps.append(_gaussian_heatmap(junction, sigma))
        degree_maps.append(degree_map)
    return (
        torch.from_numpy(np.stack(heatmaps)[:, None]).to(
            device=wall.device,
            dtype=torch.float32,
        ),
        torch.from_numpy(np.stack(degree_maps)).to(
            device=wall.device,
            dtype=torch.long,
        ),
    )


def opening_center_mask(labels: np.ndarray, taxonomy: FloorplanTaxonomy) -> np.ndarray:
    mask = np.isin(labels, np.asarray(taxonomy.opening_ids))
    components, count = component_label(mask)
    centers = np.zeros_like(mask, dtype=np.uint8)
    for component_id in range(1, int(count) + 1):
        ys, xs = np.nonzero(components == component_id)
        if len(xs):
            centers[round(ys.mean()), round(xs.mean())] = 1
    return centers


def build_geometry_targets(
    labels: torch.Tensor,
    footprint: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
    *,
    sigma: float = 1.5,
    centerline_sigma: float = 1.0,
) -> dict[str, torch.Tensor]:
    labels_array = labels.detach().cpu().numpy()
    footprint_array = footprint.detach().cpu().numpy()
    batch_targets = {key: [] for key in HEATMAP_KEYS}
    for drawing, area in zip(labels_array, footprint_array):
        wall = drawing == taxonomy.class_id("wall")
        opening = np.isin(drawing, np.asarray(taxonomy.opening_ids))
        wall_graph = wall | opening
        closed = cv2.morphologyEx(
            wall_graph.astype(np.uint8),
            cv2.MORPH_CLOSE,
            np.ones((3, 3), dtype=np.uint8),
        )
        centerline = skeletonize(closed > 0)
        junction = _junction_mask(centerline)
        opening_center = opening_center_mask(drawing, taxonomy)
        rooms = np.isin(drawing, np.asarray(taxonomy.room_ids))
        room_boundary = rooms & ~binary_erosion(
            rooms,
            structure=np.ones((3, 3), dtype=bool),
            border_value=0,
        )
        room_boundary &= np.asarray(area, dtype=bool)
        batch_targets["junction"].append(_gaussian_heatmap(junction, sigma))
        batch_targets["wall_centerline"].append(
            _gaussian_heatmap(centerline, centerline_sigma)
        )
        batch_targets["opening_center"].append(
            _gaussian_heatmap(opening_center, sigma)
        )
        batch_targets["room_boundary"].append(
            _gaussian_heatmap(room_boundary, centerline_sigma)
        )
    return {
        key: torch.from_numpy(np.stack(values)[:, None]).to(
            device=labels.device,
            dtype=torch.float32,
        )
        for key, values in batch_targets.items()
    }


def structural_candidate_mask(
    labels: torch.Tensor,
    footprint: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
    *,
    radius: int = 2,
) -> torch.Tensor:
    """Mark room transitions and the envelope as valid structural locations."""

    room_support = torch.isin(
        labels,
        torch.tensor(taxonomy.room_ids, device=labels.device),
    )
    room_labels = torch.where(room_support, labels, torch.zeros_like(labels))
    horizontal = torch.zeros_like(room_labels, dtype=torch.bool)
    vertical = torch.zeros_like(room_labels, dtype=torch.bool)
    horizontal[:, :, 1:] = room_labels[:, :, 1:] != room_labels[:, :, :-1]
    vertical[:, 1:, :] = room_labels[:, 1:, :] != room_labels[:, :-1, :]
    transitions = horizontal | vertical
    area = footprint.bool()
    if area.ndim == 2:
        area = area[None]
    outside = F.pad(
        (~area).float()[:, None],
        (1, 1, 1, 1),
        mode="constant",
        value=1.0,
    )
    exterior = area & (
        F.max_pool2d(outside, kernel_size=3, stride=1)[:, 0] > 0.5
    )
    candidates = (transitions | exterior)[:, None].float()
    if radius > 0:
        candidates = F.max_pool2d(
            candidates,
            kernel_size=2 * int(radius) + 1,
            stride=1,
            padding=int(radius),
        )
    return candidates[:, 0] > 0.5


def focal_heatmap_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    alpha: float = 2.0,
    beta: float = 4.0,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    expected = target.float().clamp(0.0, 1.0)
    probability = logits.sigmoid().clamp(1.0e-4, 1.0 - 1.0e-4)
    positive = (
        expected
        * (1.0 - probability).pow(alpha)
        * probability.log()
    )
    negative = (
        (1.0 - expected).pow(beta)
        * probability.pow(alpha)
        * (1.0 - probability).log()
    )
    if valid_mask is None:
        positive_count = expected.sum().clamp_min(1.0)
        negative_count = (1.0 - expected).sum().clamp_min(1.0)
        return -positive.sum() / positive_count - negative.sum() / negative_count
    valid = valid_mask.to(expected.dtype)
    positive_count = (expected * valid).sum().clamp_min(1.0)
    negative_count = ((1.0 - expected) * valid).sum().clamp_min(1.0)
    return -(positive * valid).sum() / positive_count - (negative * valid).sum() / negative_count


def geometry_heatmap_loss(
    predictions: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    *,
    opening_weight: float = 2.0,
    boundary_weight: float = 0.5,
    structural_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    weights = {
        "junction": 1.0,
        "wall_centerline": 1.0,
        "opening_center": float(opening_weight),
        "room_boundary": float(boundary_weight),
    }
    losses = {}
    for name in HEATMAP_KEYS:
        valid_mask = (
            structural_mask
            if name in {"junction", "wall_centerline", "opening_center"}
            else None
        )
        losses[name] = focal_heatmap_loss(
            predictions[f"{name}_logits"],
            targets[name],
            valid_mask=valid_mask,
        )
    total = sum(weights[name] * losses[name] for name in HEATMAP_KEYS)
    return total, {
        **{f"geometry_{name}": value.detach() for name, value in losses.items()},
        "geometry_total": total.detach(),
    }


def opening_candidate_loss(
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    footprint: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
    *,
    radius: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Focus opening learning on valid room-transition and envelope bands."""

    candidates = structural_candidate_mask(
        labels,
        footprint,
        taxonomy,
        radius=radius,
    ).float()
    foreground_logits = torch.logsumexp(outputs["opening_logits"], dim=1)
    binary_logits = foreground_logits - outputs["opening_background_logits"]
    expected = torch.isin(
        labels,
        torch.tensor(taxonomy.opening_ids, device=labels.device),
    ).float()
    probability = binary_logits.sigmoid().clamp(1.0e-4, 1.0 - 1.0e-4)
    positive = expected * (1.0 - probability).pow(2.0) * probability.log()
    negative = (
        (1.0 - expected).pow(4.0)
        * probability.pow(2.0)
        * (1.0 - probability).log()
    )
    loss = -((positive + negative) * candidates).sum() / candidates.sum().clamp_min(1.0)
    return loss, loss.detach()


def predicted_geometry_masks(
    predictions: dict[str, torch.Tensor],
    threshold: float = 0.5,
) -> dict[str, torch.Tensor]:
    return {
        name: predictions[f"{name}_logits"].sigmoid() >= float(threshold)
        for name in HEATMAP_KEYS
    }
