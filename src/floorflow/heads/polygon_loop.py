"""Query-based discrete room-loop prediction and Hungarian polygon loss."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.room_metric_suite import matched_instance_metrics
from floorflow.geometry.diff_ras import diff_soft_polygon_masks
from floorflow.objectives.gpu_assignment import assignment_index_tensors
from floorflow.objectives.room_query_matching import (
    RoomQueryMatch,
    match_room_queries,
)
from floorflow.representation.polygon_targets import (
    collapse_invalid_polygon_points,
    rasterize_room_polygons,
)


def _unit_loop(points: int) -> torch.Tensor:
    angles = torch.arange(points, dtype=torch.float32) * (2.0 * math.pi / points)
    return torch.stack(
        [
            0.5 + 0.25 * torch.cos(angles),
            0.5 + 0.25 * torch.sin(angles),
        ],
        dim=-1,
    )


class PolygonLoopHead(nn.Module):
    """Predict room class, normalized closed loop, and validity per query."""

    def __init__(
        self,
        in_channels: int,
        num_room_classes: int,
        hidden_dim: int = 128,
        num_queries: int = 24,
        points: int = 24,
        layers: int = 2,
        heads: int = 4,
        class_token_dim: int = 96,
    ) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.points = int(points)
        self.project = nn.Conv2d(in_channels, hidden_dim, 1)
        self.queries = nn.Parameter(torch.randn(1, num_queries, hidden_dim) * 0.02)
        self.corner_queries = nn.Parameter(
            torch.randn(1, self.points, hidden_dim) * 0.02
        )
        self.class_tokens = nn.Parameter(
            torch.randn(1, num_room_classes, hidden_dim) * 0.02
        )
        self.class_projection = (
            nn.Linear(class_token_dim, hidden_dim)
            if class_token_dim != hidden_dim
            else nn.Identity()
        )
        self.layers = nn.ModuleList(
            [
                nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
                for _ in range(max(1, int(layers)))
            ]
        )
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.memory_norm = nn.LayerNorm(hidden_dim)
        self.corner_norm = nn.LayerNorm(hidden_dim)
        self.corner_attention = nn.MultiheadAttention(
            hidden_dim,
            heads,
            batch_first=True,
        )
        self.class_head = nn.Linear(hidden_dim, num_room_classes)
        self.center_head = nn.Linear(hidden_dim, 2)
        self.corner_head = nn.Linear(
            self.points * hidden_dim,
            self.points * 2,
        )
        self.valid_head = nn.Linear(hidden_dim, 1)
        self.register_buffer("base_loop", _unit_loop(self.points))
        nn.init.zeros_(self.corner_head.weight)
        nn.init.zeros_(self.corner_head.bias)
        nn.init.normal_(self.valid_head.weight, std=0.01)
        nn.init.constant_(self.valid_head.bias, 1.0)

    def forward(
        self,
        features: torch.Tensor,
        class_tokens: torch.Tensor | None = None,
        query_prior: torch.Tensor | None = None,
        extra_memory: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        memory = self.project(features)
        memory = F.adaptive_avg_pool2d(memory, (16, 16))
        memory = self.memory_norm(memory.flatten(2).transpose(1, 2))
        if extra_memory is not None:
            memory = torch.cat([memory, extra_memory.to(memory.dtype)], dim=1)
        query = self.queries.to(features.dtype).expand(features.shape[0], -1, -1)
        if class_tokens is not None:
            class_tokens = self.class_projection(class_tokens.to(query.dtype))
            query = query + F.adaptive_avg_pool1d(
                class_tokens.transpose(1, 2),
                1,
            ).transpose(1, 2)
        if query_prior is not None:
            if query_prior.shape != query.shape:
                raise ValueError("query_prior must match [B,Q,D]")
            query = query + query_prior.to(query.dtype)
        for attention in self.layers:
            attended, _weights = attention(
                self.query_norm(query),
                memory,
                memory,
                need_weights=False,
            )
            query = query + attended
        center = torch.sigmoid(self.center_head(query))
        batch, query_count = query.shape[:2]
        corner_tokens = self.corner_queries.to(query.dtype) + query[:, :, None]
        corner_tokens = self.corner_norm(corner_tokens)
        expanded_memory = memory[:, None].expand(
            -1,
            query_count,
            -1,
            -1,
        ).reshape(batch * query_count, memory.shape[1], memory.shape[2])
        attended_corners, _corner_weights = self.corner_attention(
            corner_tokens.reshape(batch * query_count, self.points, -1),
            expanded_memory,
            expanded_memory,
            need_weights=False,
        )
        corner_context = corner_tokens + attended_corners.reshape(
            batch,
            query_count,
            self.points,
            -1,
        )
        delta = self.corner_head(corner_context.reshape(
            batch,
            query_count,
            self.points * corner_context.shape[-1],
        )).reshape(
            query.shape[0],
            query.shape[1],
            self.points,
            2,
        )
        polygons = center[:, :, None, :] + self.base_loop.to(query.dtype)[None, None]
        polygons = polygons + 0.25 * torch.tanh(delta)
        return {
            "class_logits": self.class_head(query),
            "polygons": polygons.clamp(0.0, 1.0),
            "valid_logits": self.valid_head(query).squeeze(-1),
            "corner_attention": _corner_weights,
            "query_hidden": query,
        }


@torch.no_grad()
def room_query_metrics(
    predictions: dict[str, torch.Tensor],
    target: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
    *,
    validity_threshold: float = 0.5,
    min_area: int = 8,
    query_budget: torch.Tensor | None = None,
    query_quality_logits: torch.Tensor | None = None,
    query_quality_weight: float = 0.5,
) -> dict[str, float | int]:
    height, width = target.shape[-2:]
    classes = predictions["class_logits"].argmax(dim=-1)
    polygons = predictions["polygons"]
    if "room_point_valid_logits" in predictions:
        polygons = collapse_invalid_polygon_points(
            polygons,
            predictions["room_point_valid_logits"] >= 0.0,
        )
    score = predictions["valid_logits"].sigmoid()
    valid = score >= float(validity_threshold)
    if query_budget is not None:
        budgets = query_budget.to(score.device).long().clamp(0, score.shape[1])
        ranking_score = predictions["valid_logits"]
        if query_quality_logits is not None:
            ranking_score = (
                ranking_score
                + float(query_quality_weight) * query_quality_logits
            )
        ranks = ranking_score.argsort(dim=-1, descending=True).argsort(dim=-1)
        valid = ranks < budgets[:, None]
    rendered = rasterize_room_polygons(
        polygons,
        classes,
        valid,
        taxonomy,
        height=height,
        width=width,
    )
    rows = []
    for prediction, expected, predicted_valid in zip(
        rendered.cpu().numpy(),
        target.cpu().numpy(),
        valid,
    ):
        rows.append(
            matched_instance_metrics(
                prediction,
                expected,
                taxonomy,
                min_area=min_area,
            )
        )
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in (
            "count_abs_error",
            "matched_geometric_iou",
            "instance_f1_at_threshold",
            "matched_type_accuracy",
        )
    }


def polygon_loop_loss(
    predictions: dict[str, torch.Tensor],
    target_classes: torch.Tensor,
    target_polygons: torch.Tensor,
    target_valid: torch.Tensor,
    *,
    class_weight: float = 1.0,
    polygon_weight: float = 2.0,
    chamfer_weight: float = 0.5,
    mask_dice_weight: float = 0.25,
    valid_weight: float = 0.5,
    valid_positive_weight: float = 1.0,
    mask_grid: int = 24,
    mask_rasterizer: str = "diff_ras",
    matches: list[RoomQueryMatch] | None = None,
    target_point_valid: torch.Tensor | None = None,
    point_validity_weight: float = 0.5,
    target_counts: Sequence[int] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Class-aware Hungarian matching followed by loop regression."""

    class_losses = []
    polygon_losses = []
    chamfer_losses = []
    mask_dice_losses = []
    valid_losses = []
    point_validity_losses = []
    matches = matches or match_room_queries(
        predictions["class_logits"],
        predictions["polygons"],
        target_classes,
        target_polygons,
        target_valid,
        target_counts=target_counts,
    )
    assignment_pairs = [
        (match.predicted_indices, match.target_indices)
        for match in matches
    ]
    query_rows, query_targets, query_match_valid = assignment_index_tensors(
        assignment_pairs,
        device=predictions["valid_logits"].device,
    )
    counts = (
        [int(value) for value in target_counts]
        if target_counts is not None
        else [int(value) for value in target_valid.sum(dim=1).tolist()]
    )
    valid_positive_weight = predictions["valid_logits"].new_tensor(
        float(valid_positive_weight)
    )
    for index in range(target_classes.shape[0]):
        valid = target_valid[index]
        count = counts[index]
        valid_target = valid.float()
        valid_losses.append(
            F.binary_cross_entropy_with_logits(
                predictions["valid_logits"][index],
                valid_target,
                pos_weight=valid_positive_weight,
            )
        )
        if count == 0:
            continue
        predicted_index = query_rows[index, query_match_valid[index]]
        target_index = query_targets[index, query_match_valid[index]]
        class_losses.append(
            F.cross_entropy(
                predictions["class_logits"][index, predicted_index],
                target_classes[index, target_index],
            )
        )
        polygon_losses.append(
            F.l1_loss(
                predictions["polygons"][index, predicted_index],
                target_polygons[index, target_index],
            )
        )
        matched_predictions = predictions["polygons"][index, predicted_index]
        matched_targets = target_polygons[index, target_index]
        chamfer_losses.append(
            _chamfer_distance(matched_predictions, matched_targets).mean()
        )
        predicted_masks = _soft_polygon_masks(
            matched_predictions,
            mask_grid,
            rasterizer=mask_rasterizer,
        )
        target_masks = _soft_polygon_masks(
            matched_targets,
            mask_grid,
            rasterizer=mask_rasterizer,
        )
        mask_intersection = (predicted_masks * target_masks).flatten(1).sum(dim=1)
        mask_denominator = (
            predicted_masks + target_masks
        ).flatten(1).sum(dim=1)
        mask_dice_losses.append(
            (
                1.0
                - (2.0 * mask_intersection + 1.0)
                / (mask_denominator + 1.0)
            ).mean()
        )
        matched_valid = torch.zeros_like(predictions["valid_logits"][index])
        matched_valid[predicted_index] = 1.0
        valid_losses.append(
            F.binary_cross_entropy_with_logits(
                predictions["valid_logits"][index],
                matched_valid,
                pos_weight=valid_positive_weight,
            )
        )
        if target_point_valid is not None:
            if "room_point_valid_logits" not in predictions:
                raise ValueError(
                    "target point validity requires a point validity head"
                )
            predicted_points = predictions["room_point_valid_logits"][
                index,
                predicted_index,
            ]
            expected_points = target_point_valid[index, target_index]
            point_validity_losses.append(
                F.binary_cross_entropy_with_logits(
                    predicted_points,
                    expected_points.float(),
                )
            )
    class_loss = sum(class_losses) / max(1, len(class_losses))
    polygon_loss = sum(polygon_losses) / max(1, len(polygon_losses))
    chamfer_loss = sum(chamfer_losses) / max(1, len(chamfer_losses))
    mask_dice_loss = sum(mask_dice_losses) / max(1, len(mask_dice_losses))
    valid_loss = sum(valid_losses) / max(1, len(valid_losses))
    point_validity_loss = (
        sum(point_validity_losses) / max(1, len(point_validity_losses))
        if point_validity_losses
        else valid_loss * 0.0
    )
    total = (
        float(class_weight) * class_loss
        + float(polygon_weight) * polygon_loss
        + float(chamfer_weight) * chamfer_loss
        + float(mask_dice_weight) * mask_dice_loss
        + float(valid_weight) * valid_loss
        + float(point_validity_weight) * point_validity_loss
    )
    return total, {
        "room_polygon_loss": total.detach(),
        "room_polygon_class_loss": class_loss.detach(),
        "room_polygon_corner_loss": polygon_loss.detach(),
        "room_polygon_chamfer_loss": chamfer_loss.detach(),
        "room_polygon_mask_dice_loss": mask_dice_loss.detach(),
        "room_polygon_valid_loss": valid_loss.detach(),
        "room_polygon_point_valid_loss": point_validity_loss.detach(),
    }


def _chamfer_distance(
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    distances = torch.cdist(left, right)
    return 0.5 * (
        distances.min(dim=2).values.mean(dim=1)
        + distances.min(dim=1).values.mean(dim=1)
    )


def _soft_polygon_masks(
    polygons: torch.Tensor,
    grid_size: int,
    *,
    rasterizer: str = "diff_ras",
) -> torch.Tensor:
    if rasterizer == "diff_ras":
        return diff_soft_polygon_masks(polygons, grid_size)
    if rasterizer != "winding":
        raise ValueError(f"unknown polygon rasterizer: {rasterizer}")
    coordinates = torch.linspace(
        0.0,
        1.0,
        int(grid_size),
        device=polygons.device,
        dtype=polygons.dtype,
    )
    grid_y, grid_x = torch.meshgrid(
        coordinates,
        coordinates,
        indexing="ij",
    )
    points = torch.stack([grid_x, grid_y], dim=-1)[None, None]
    vertices = polygons[:, :, None, None, :]
    following = torch.roll(polygons, shifts=-1, dims=1)[:, :, None, None, :]
    first = points - vertices
    second = points - following
    cross = first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]
    dot = (first * second).sum(dim=-1)
    winding = torch.atan2(cross, dot).sum(dim=2) / (2.0 * math.pi)
    return torch.sigmoid((winding.abs() - 0.5) * 20.0)
