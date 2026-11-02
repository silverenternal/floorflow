"""Dedicated foreground and door/window heatmap head."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.geometry.opening_readout import detections_from_labels
from floorflow.geometry.wall_graph_readout import raster_labels_to_wall_plan
from floorflow.heads.geometry_heads import focal_heatmap_loss
from floorflow.heads.room_segmenter import ConvNormAct
from floorflow.geometry.deterministic_pooling import fixed_grid_pool
from floorflow.model.official_mita_route import OfficialMiTARouteAttention
from floorflow.objectives.semantic_losses import dice_loss, focal_loss


class OpeningDetectionHead(nn.Module):
    """Predict one opening mask and a two-family classification map."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 96,
        variant: str = "v1",
        spatial_gap_attention: bool = False,
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim)
        if variant not in {"v1", "multiscale", "center_width"}:
            raise ValueError("unsupported opening head variant")
        self.variant = variant
        self.stem = nn.Sequential(
            ConvNormAct(int(in_channels), hidden_dim),
            ConvNormAct(hidden_dim, hidden_dim),
        )
        if variant in {"multiscale", "center_width"}:
            self.multiscale = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(
                            hidden_dim,
                            hidden_dim,
                            3,
                            padding=dilation,
                            dilation=dilation,
                            bias=False,
                        ),
                        nn.GroupNorm(max(1, hidden_dim // 16), hidden_dim),
                        nn.SiLU(),
                    )
                    for dilation in (1, 2, 4)
                ]
            )
            self.multiscale_fusion = nn.Sequential(
                nn.Conv2d(hidden_dim * 3, hidden_dim, 1, bias=False),
                nn.GroupNorm(max(1, hidden_dim // 16), hidden_dim),
                nn.SiLU(),
            )
            self.family_context = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
                nn.GroupNorm(max(1, hidden_dim // 16), hidden_dim),
                nn.SiLU(),
            )
        self.foreground = nn.Conv2d(hidden_dim, 1, 1)
        self.family = nn.Conv2d(hidden_dim, 2, 1)
        self.center = (
            nn.Conv2d(hidden_dim, 1, 1)
            if variant == "center_width"
            else None
        )
        self.width = (
            nn.Conv2d(hidden_dim, 1, 1)
            if variant == "center_width"
            else None
        )
        self.gap_fusion = None
        if variant == "center_width":
            self.gap_tokens = nn.Parameter(torch.zeros(4, hidden_dim))
            self.gap_context = nn.Linear(hidden_dim, hidden_dim)
            self.gap_attention = nn.MultiheadAttention(
                hidden_dim,
                max(1, hidden_dim // 16),
                batch_first=True,
            )
            self.spatial_gap_attention = (
                OfficialMiTARouteAttention(
                    hidden_dim,
                    num_heads=max(1, hidden_dim // 16),
                    router_count=min(8, 16 * 16),
                    top_k=min(4, 16 * 16),
                )
                if spatial_gap_attention
                else None
            )
            self.gap_memory_norm = (
                nn.LayerNorm(hidden_dim)
                if spatial_gap_attention
                else None
            )
            self.gap_norm = nn.LayerNorm(hidden_dim)
            self.gap_gates = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Sigmoid(),
            )
            self.gap_scale = nn.Parameter(torch.tensor(0.01))
            self.gap_fusion = True
        with torch.no_grad():
            self.foreground.bias.fill_(-5.0)
            self.family.weight.mul_(0.01)
            self.family.bias.zero_()
            if self.center is not None:
                self.center.bias.fill_(-5.0)
            if self.width is not None:
                self.width.weight.mul_(0.01)
                self.width.bias.fill_(-2.0)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.stem(features)
        if self.variant in {"multiscale", "center_width"}:
            hidden = self.multiscale_fusion(
                torch.cat(
                    [branch(hidden) for branch in self.multiscale],
                    dim=1,
                )
            )
            family_hidden = self.family_context(hidden)
        else:
            family_hidden = hidden
        gap_diagnostics = None
        if self.gap_fusion:
            context = hidden.mean(dim=(-2, -1))
            tokens = self.gap_tokens[None] + self.gap_context(context)[:, None]
            tokens = self.gap_norm(
                tokens
                + self.gap_attention(tokens, tokens, tokens, need_weights=False)[0]
            )
            if (
                self.spatial_gap_attention is not None
                and self.gap_memory_norm is not None
            ):
                memory = fixed_grid_pool(hidden, (16, 16))
                memory = self.gap_memory_norm(
                    memory.flatten(2).transpose(1, 2)
                )
                spatial_context, gap_diagnostics = self.spatial_gap_attention(
                    tokens,
                    memory,
                )
                tokens = self.gap_norm(tokens + spatial_context)
            gates = self.gap_gates(tokens).reshape(
                hidden.shape[0],
                4,
                hidden.shape[1],
            )
            modulation = 1.0 + torch.tanh(self.gap_scale) * gates
            foreground_hidden = hidden * modulation[:, 0, :, None, None]
            center_hidden = hidden * modulation[:, 1, :, None, None]
            width_hidden = family_hidden * modulation[:, 3, :, None, None]
            family_hidden = family_hidden * modulation[:, 2, :, None, None]
        else:
            foreground_hidden = hidden
            center_hidden = hidden
            width_hidden = family_hidden
        outputs = {
            "opening_logits": self.foreground(foreground_hidden),
            "family_logits": self.family(family_hidden),
        }
        if self.center is not None and self.width is not None:
            outputs["opening_center_logits"] = self.center(center_hidden)
            outputs["opening_width_logits"] = self.width(width_hidden)
        if self.gap_fusion and gap_diagnostics is not None:
            outputs["gap_route_entropy"] = gap_diagnostics[
                "routing_entropy"
            ]
            outputs["gap_attention_entropy"] = gap_diagnostics[
                "attention_entropy"
            ]
        return outputs


def opening_center_width_targets(
    target: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
    *,
    center_radius: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build peak-location and interval-width maps from labeled openings."""

    labels = target.detach().cpu().numpy()
    batch, height, width = labels.shape
    center = torch.zeros(batch, 1, height, width, dtype=torch.float32)
    interval_width = torch.zeros_like(center)
    for index in range(batch):
        plan = raster_labels_to_wall_plan(labels[index], taxonomy)
        detections = detections_from_labels(labels[index], taxonomy, plan)
        for detection in detections:
            center_x = min(max(round(detection.center[0]), 0), width - 1)
            center_y = min(max(round(detection.center[1]), 0), height - 1)
            for y in range(
                max(0, center_y - center_radius),
                min(height, center_y + center_radius + 1),
            ):
                for x in range(
                    max(0, center_x - center_radius),
                    min(width, center_x + center_radius + 1),
                ):
                    distance_sq = (x - center_x) ** 2 + (y - center_y) ** 2
                    center[index, 0, y, x] = max(
                        float(center[index, 0, y, x]),
                        math.exp(-distance_sq / 2.0),
                    )
            interval_width[index, 0, center_y, center_x] = (
                float(detection.width) / float(width)
            )
    return center.to(target.device), interval_width.to(target.device)


def opening_detection_loss(
    outputs: dict[str, torch.Tensor],
    target: torch.Tensor,
    footprint: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
    *,
    foreground_weight: float = 1.0,
    dice_weight: float = 1.0,
    family_weight: float = 0.5,
    wall_band_weight: float = 0.5,
    wall_band_radius: int = 4,
    center_weight: float = 0.5,
    width_weight: float = 0.25,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Dense opening loss with extra supervision near reference walls."""

    opening_ids = torch.tensor(
        taxonomy.opening_ids,
        device=target.device,
    )
    door_ids = torch.tensor(
        [
            class_id
            for class_id in taxonomy.opening_ids
            if taxonomy.class_names[class_id].startswith("door:")
        ],
        device=target.device,
    )
    opening = torch.isin(target, opening_ids)
    door = torch.isin(target, door_ids)
    valid = footprint.bool()
    foreground = outputs["opening_logits"][:, 0]
    focal = focal_loss(foreground[valid], opening[valid].float())
    dice = dice_loss(
        foreground.masked_fill(~valid, -20.0),
        opening.float() * valid.float(),
    )
    family_logits = outputs["family_logits"].permute(0, 2, 3, 1)[opening]
    family_target = (~door[opening]).long()
    family = (
        F.cross_entropy(family_logits, family_target)
        if family_logits.numel()
        else foreground.sum() * 0.0
    )
    wall_id = taxonomy.class_id("wall")
    wall_band = (
        ((target == wall_id) | opening).float()[:, None]
    )
    radius = max(0, int(wall_band_radius))
    if radius:
        wall_band = F.max_pool2d(
            wall_band,
            2 * radius + 1,
            stride=1,
            padding=radius,
        )
    band_valid = valid & (wall_band[:, 0] > 0.5)
    if band_valid.any():
        band_logits = foreground[band_valid]
        band_target = opening[band_valid].float()
        band_focal = F.binary_cross_entropy_with_logits(
            band_logits,
            band_target,
        )
        probability = band_logits.sigmoid()
        true_positive = (probability * band_target).sum()
        false_positive = (probability * (1.0 - band_target)).sum()
        false_negative = ((1.0 - probability) * band_target).sum()
        band_tversky = 1.0 - (
            true_positive + 1.0
        ) / (
            true_positive
            + 0.6 * false_positive
            + 0.4 * false_negative
            + 1.0
        )
    else:
        band_focal = foreground.sum() * 0.0
        band_tversky = foreground.sum() * 0.0
    wall_band_loss = band_focal + band_tversky
    total = (
        float(foreground_weight) * focal
        + float(dice_weight) * dice
        + float(family_weight) * family
        + float(wall_band_weight) * wall_band_loss
    )
    center_loss = foreground.sum() * 0.0
    width_loss = foreground.sum() * 0.0
    if (
        "opening_center_logits" in outputs
        and "opening_width_logits" in outputs
    ):
        center_target, width_target = opening_center_width_targets(
            target,
            taxonomy,
        )
        center_loss = focal_heatmap_loss(
            outputs["opening_center_logits"],
            center_target,
        )
        positive = center_target[:, 0] > 0.5
        predicted_width = outputs["opening_width_logits"][:, 0].sigmoid()
        width_loss = (
            F.smooth_l1_loss(
                predicted_width[positive],
                width_target[:, 0][positive],
            )
            if positive.any()
            else predicted_width.sum() * 0.0
        )
        total = (
            total
            + float(center_weight) * center_loss
            + float(width_weight) * width_loss
        )
    components = {
        "opening_foreground_loss": focal.detach(),
        "opening_dice_loss": dice.detach(),
        "opening_family_loss": family.detach(),
        "opening_wall_band_focal": band_focal.detach(),
        "opening_wall_band_tversky": band_tversky.detach(),
    }
    if (
        "opening_center_logits" in outputs
        and "opening_width_logits" in outputs
    ):
        components["opening_center_loss"] = center_loss.detach()
        components["opening_width_loss"] = width_loss.detach()
    return total, components


def opening_detection_probabilities(
    outputs: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Return door/window probabilities as `[B,2,H,W]`."""

    foreground = outputs["opening_logits"].sigmoid()
    family = outputs["family_logits"].softmax(dim=1)
    return foreground * family


__all__ = [
    "OpeningDetectionHead",
    "opening_center_width_targets",
    "opening_detection_loss",
    "opening_detection_probabilities",
]
