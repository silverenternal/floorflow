"""Target-free per-query quality evidence for RPLAN room arbitration."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from floorflow.geometry.diff_ras import diff_soft_polygon_masks
from floorflow.recursion.trust_gate import RoomGraphTrustGate

RPLAN_QUERY_QUALITY_FEATURE_NAMES = (
    "class_confidence",
    "class_margin",
    "class_entropy",
    "existence_probability",
    "mask_mass",
    "mask_peak",
    "mask_area_fraction",
    "polygon_area",
    "polygon_mask_dice",
    "count_rank",
    "adjacency_degree",
    "routing_entropy",
)


def _polygon_area(polygons: torch.Tensor) -> torch.Tensor:
    following = torch.roll(polygons, shifts=-1, dims=2)
    cross = (
        polygons[..., 0] * following[..., 1]
        - following[..., 0] * polygons[..., 1]
    )
    return 0.5 * cross.sum(dim=-1).abs()


def rplan_query_quality_features(
    predictions: dict[str, torch.Tensor],
    *,
    polygon_grid: int = 16,
) -> torch.Tensor:
    """Build per-query quality features without any target-derived input."""

    class_logits = predictions["room_instance_class_logits"][..., :-1]
    foreground = class_logits.softmax(dim=-1)
    confidence = foreground.amax(dim=-1)
    top_two = foreground.topk(2, dim=-1).values
    margin = top_two[..., 0] - top_two[..., 1]
    entropy = -(
        foreground.clamp_min(1.0e-8) * foreground.clamp_min(1.0e-8).log()
    ).sum(dim=-1) / math.log(max(2, foreground.shape[-1]))
    existence = predictions["room_exists"].sigmoid()
    mask_probability = predictions["room_instance_mask_logits"].sigmoid()
    mask_mass = mask_probability.mean(dim=(-1, -2))
    mask_peak = mask_probability.amax(dim=(-1, -2))
    mask_area = (mask_probability >= 0.5).float().mean(dim=(-1, -2))
    polygons = predictions["room_polygons"]
    flat_polygons = polygons.flatten(0, 1)
    rendered = diff_soft_polygon_masks(flat_polygons, polygon_grid).reshape(
        *polygons.shape[:2],
        polygon_grid,
        polygon_grid,
    )
    resized_mask = F.interpolate(
        mask_probability.flatten(0, 1)[:, None],
        size=(polygon_grid, polygon_grid),
        mode="bilinear",
        align_corners=False,
    ).reshape(
        mask_probability.shape[0],
        mask_probability.shape[1],
        polygon_grid,
        polygon_grid,
    )
    intersection = (rendered * resized_mask).flatten(2).sum(dim=-1)
    denominator = rendered.flatten(2).sum(dim=-1) + resized_mask.flatten(2).sum(dim=-1)
    polygon_mask_dice = 2.0 * intersection / denominator.clamp_min(1.0e-6)
    rank = existence.argsort(dim=-1, descending=True).argsort(dim=-1).float()
    rank = rank / max(1, existence.shape[1] - 1)
    adjacency = predictions["room_adjacency_logits"].sigmoid()
    pair_mask = torch.isfinite(adjacency) & ~torch.eye(
        adjacency.shape[-1],
        device=adjacency.device,
        dtype=torch.bool,
    )[None]
    adjacency_degree = (
        adjacency * pair_mask
    ).sum(dim=-1) / pair_mask.sum(dim=-1).clamp_min(1)
    routing = predictions.get("routing")
    if routing is None:
        routing_entropy = existence.new_zeros(existence.shape)
    else:
        value = routing["routing_entropy"]
        value = value.to(existence.device, dtype=existence.dtype)
        routing_entropy = value.reshape(-1)[0].expand_as(existence)
    return torch.stack(
        [
            confidence,
            margin,
            entropy,
            existence,
            mask_mass,
            mask_peak,
            mask_area,
            _polygon_area(polygons),
            polygon_mask_dice,
            rank,
            adjacency_degree,
            routing_entropy,
        ],
        dim=-1,
    )


def make_rplan_query_quality_head(
    *,
    hidden_size: int = 32,
    device: torch.device | str | None = None,
) -> RoomGraphTrustGate:
    head = RoomGraphTrustGate(
        feature_size=len(RPLAN_QUERY_QUALITY_FEATURE_NAMES),
        hidden_size=hidden_size,
    )
    return head.to(device) if device is not None else head


def score_rplan_query_quality(
    head: RoomGraphTrustGate,
    features: torch.Tensor,
) -> torch.Tensor:
    if features.ndim != 3:
        raise ValueError("query quality features must be [B,Q,F]")
    flat = features.flatten(0, 1)
    return head(flat).reshape(features.shape[:2])


def select_quality_queries(
    existence_logits: torch.Tensor,
    quality_logits: torch.Tensor,
    query_budget: torch.Tensor,
    *,
    quality_weight: float = 0.5,
    minimum_quality: float = 0.2,
) -> torch.Tensor:
    """Select the predicted count by existence and learned query quality."""

    if existence_logits.ndim != 2 or quality_logits.shape != existence_logits.shape:
        raise ValueError(
            "existence and quality logits must share [B,Q]"
        )
    if query_budget.ndim != 1 or query_budget.shape[0] != existence_logits.shape[0]:
        raise ValueError("query budget must be [B]")
    score = existence_logits + float(quality_weight) * quality_logits
    selected = torch.zeros_like(score, dtype=torch.bool)
    budgets = query_budget.to(score.device).long().clamp(0, score.shape[1])
    indices = score.topk(int(score.shape[1]), dim=-1).indices
    for batch, budget in enumerate(budgets.tolist()):
        candidates = indices[batch, :budget]
        reliable = quality_logits[batch, candidates].sigmoid() >= float(minimum_quality)
        if budget > 0 and not bool(reliable.any()):
            reliable[0] = True
        selected[batch, candidates[reliable]] = True
    return selected


__all__ = [
    "RPLAN_QUERY_QUALITY_FEATURE_NAMES",
    "make_rplan_query_quality_head",
    "rplan_query_quality_features",
    "score_rplan_query_quality",
    "select_quality_queries",
]
