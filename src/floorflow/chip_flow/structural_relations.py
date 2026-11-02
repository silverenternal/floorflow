"""Grouping-aligned structural relation features and prediction head."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from floorflow.chip_flow.hierarchy_structure import structural_membership


def structural_relation_features(
    boxes: torch.Tensor,
    areas: torch.Tensor,
    constraints: torch.Tensor,
    *,
    adjacency_gap: float = 1.0e-3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-slot geometry/connectivity features and active masks."""

    membership = structural_membership(constraints)
    scale = areas.sum(dim=1).sqrt().clamp_min(1.0e-6)
    x0, y0 = boxes[..., 0], boxes[..., 1]
    x1, y1 = x0 + boxes[..., 2], y0 + boxes[..., 3]
    gap_x = torch.maximum(
        x0[:, :, None],
        x0[:, None, :],
    ) - torch.minimum(x1[:, :, None], x1[:, None, :])
    gap_y = torch.maximum(
        y0[:, :, None],
        y0[:, None, :],
    ) - torch.minimum(y1[:, :, None], y1[:, None, :])
    gap = torch.maximum(gap_x, gap_y)
    pair_mask = membership[:, :, None, :] & membership[:, None, :, :]
    pair_mask = pair_mask & ~torch.eye(
        boxes.shape[1],
        dtype=torch.bool,
        device=boxes.device,
    )[None, :, :, None]
    weight = pair_mask.to(boxes.dtype)
    count = weight.sum(dim=(1, 2)).clamp_min(1.0)
    normalized_gap = gap / scale[:, None, None]
    mean_gap = (normalized_gap[..., None] * weight).sum(dim=(1, 2)) / count
    adjacent = pair_mask & normalized_gap[..., None].le(
        float(adjacency_gap)
    )
    member_count = membership.sum(dim=1).clamp_min(1.0)
    reach = adjacent | (
        torch.eye(
            boxes.shape[1],
            dtype=torch.bool,
            device=boxes.device,
        )[None, :, :, None]
        & membership[:, :, None, :]
        & membership[:, None, :, :]
    )
    for _ in range(max(1, boxes.shape[1].bit_length())):
        reach = reach | (
            torch.einsum(
                "bijs,bjks->biks",
                reach.float(),
                reach.float(),
            ).gt(0)
        )
    positions = torch.arange(boxes.shape[1], device=boxes.device)
    earlier = positions[None, :, None] > positions[None, None, :]
    representative = membership & ~(
        reach & earlier[..., None]
    ).any(dim=2)
    connected_components = representative.sum(dim=1)
    component_deficit = (connected_components - 1).clamp_min(0) / member_count

    # A group moment must describe the group's occupied bbox. Averaging member
    # left/right edges instead measures mean member size and discards separation.
    active = membership.any(dim=1)
    x0_slot = torch.where(membership, x0[:, :, None], torch.inf).amin(dim=1)
    y0_slot = torch.where(membership, y0[:, :, None], torch.inf).amin(dim=1)
    x1_slot = torch.where(membership, x1[:, :, None], -torch.inf).amax(dim=1)
    y1_slot = torch.where(membership, y1[:, :, None], -torch.inf).amax(dim=1)
    center_x = torch.where(
        active,
        0.5 * (x0_slot + x1_slot) / scale[:, None],
        torch.zeros_like(x0_slot),
    )
    center_y = torch.where(
        active,
        0.5 * (y0_slot + y1_slot) / scale[:, None],
        torch.zeros_like(y0_slot),
    )
    width = torch.where(
        active,
        (x1_slot - x0_slot) / scale[:, None],
        torch.zeros_like(x0_slot),
    )
    height = torch.where(
        active,
        (y1_slot - y0_slot) / scale[:, None],
        torch.zeros_like(y0_slot),
    )
    area_fraction = (
        (areas[:, :, None] * membership).sum(dim=1)
        / areas.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
    )
    features = torch.stack(
        [
            center_x,
            center_y,
            width,
            height,
            area_fraction,
            mean_gap,
            component_deficit,
        ],
        dim=-1,
    )
    return features * active[..., None], active


class StructuralRelationHead(nn.Module):
    """Predict grouping-aligned relation features and contextualize L1 tokens."""

    def __init__(
        self,
        hidden: int,
        *,
        structural_slots: int = 9,
        bottleneck: int | None = None,
    ) -> None:
        super().__init__()
        if hidden <= 0 or structural_slots <= 0:
            raise ValueError("hidden and structural_slots must be positive")
        width = int(bottleneck or max(16, hidden // 8))
        self.hidden = int(hidden)
        self.structural_slots = int(structural_slots)
        self.token_encoder = nn.Linear(hidden, width)
        self.state_encoder = nn.Linear(3, width)
        self.block_state_encoder = nn.Linear(3, width)
        self.count_encoder = nn.Linear(2, width)
        self.feature_head = nn.Sequential(
            nn.Linear(4 * width, width),
            nn.SiLU(),
            nn.Linear(width, 7),
        )
        self.token_head = nn.Sequential(
            nn.Linear(4 * width, width),
            nn.SiLU(),
            nn.Linear(width, hidden),
        )
        nn.init.zeros_(self.feature_head[-1].weight)
        nn.init.zeros_(self.feature_head[-1].bias)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        block_tokens: torch.Tensor,
        block_state: torch.Tensor,
        areas: torch.Tensor,
        membership: torch.Tensor,
        structural_mask: torch.Tensor,
        structural_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, count, hidden = block_tokens.shape
        if hidden != self.hidden:
            raise ValueError("block token hidden size mismatch")
        if membership.shape != (batch, count, self.structural_slots):
            raise ValueError("membership has the wrong structural-slot count")
        if structural_state.shape != (batch, self.structural_slots, 3):
            raise ValueError("structural state has the wrong shape")
        weight = membership.to(block_tokens.dtype)
        denominator = weight.sum(dim=1).clamp_min(1.0)[..., None]
        pooled = torch.einsum("bns,bnh->bsh", weight, block_tokens)
        pooled = pooled / denominator
        pooled_state = torch.einsum("bns,bnd->bsd", weight, block_state)
        pooled_state = pooled_state / denominator
        member_count = membership.sum(dim=1)
        area_fraction = (
            (areas[:, :, None] * membership).sum(dim=1)
            / areas.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
        )
        count_features = torch.stack(
            [member_count.log1p(), area_fraction],
            dim=-1,
        )
        features = torch.cat(
            [
                self.token_encoder(pooled),
                self.state_encoder(structural_state.float()),
                self.block_state_encoder(pooled_state.float()),
                self.count_encoder(count_features),
            ],
            dim=-1,
        )
        relation_features = self.feature_head(features)
        token_residual = self.gate * self.token_head(features)
        active = structural_mask
        relation_features = relation_features * active[..., None]
        token_residual = token_residual * active[..., None]
        return relation_features, token_residual


def structural_relation_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    beta: float = 0.1,
) -> torch.Tensor:
    """Masked smooth-L1 over grouping geometry/connectivity features."""

    if prediction.shape != target.shape:
        raise ValueError("structural relation shapes must match")
    error = F.smooth_l1_loss(prediction, target, beta=beta, reduction="none")
    weight = mask[..., None].to(error.dtype).expand_as(error)
    return (error * weight).sum() / weight.sum().clamp_min(1.0)


__all__ = [
    "StructuralRelationHead",
    "structural_relation_features",
    "structural_relation_loss",
]
