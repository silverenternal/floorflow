"""Shared differentiable room-adjacency readout."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def room_adjacency_logits(
    features: torch.Tensor,
    projection: nn.Linear,
) -> torch.Tensor:
    """Return a symmetric dual-query adjacency logit matrix."""

    projected = F.normalize(projection(features), dim=-1)
    logits = torch.einsum("bqd,bkd->bqk", projected, projected)
    return 0.5 * (logits + logits.transpose(1, 2))


def mask_contact_logits(
    mask_logits: torch.Tensor,
    *,
    tolerance: int = 2,
) -> torch.Tensor:
    """Map differentiable query-mask contact into adjacency-logit space."""

    if mask_logits.ndim != 4:
        raise ValueError("mask logits must be [B,Q,H,W]")
    probability = mask_logits.float().sigmoid()
    radius = max(1, int(tolerance) + 1)
    neighbor = F.max_pool2d(
        probability,
        2 * radius + 1,
        stride=1,
        padding=radius,
    )
    intersection = torch.bmm(
        probability.flatten(2),
        neighbor.flatten(2).transpose(1, 2),
    )
    area = probability.flatten(2).sum(dim=-1)
    scale = torch.sqrt(area[:, :, None] * area[:, None, :]).clamp_min(
        1.0e-6
    )
    contact = (intersection / scale).clamp(0.0, 1.0)
    contact = 0.5 * (contact + contact.transpose(1, 2))
    diagonal = torch.eye(
        contact.shape[-1],
        device=contact.device,
        dtype=torch.bool,
    )[None]
    contact = contact.masked_fill(diagonal, 0.0)
    return torch.logit(contact.clamp(1.0e-4, 1.0 - 1.0e-4))


class RoomAdjacencyHead(nn.Module):
    """Identity-initialized adjacency projection over room query features."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        mask_aware: bool = False,
        contact_tolerance: int = 2,
        relational: bool = False,
        relational_hidden: int = 128,
    ) -> None:
        super().__init__()
        self.projection = nn.Linear(int(hidden_dim), int(hidden_dim), bias=False)
        nn.init.eye_(self.projection.weight)
        self.relational = bool(relational)
        self.mask_aware = bool(mask_aware or relational)
        self.contact_tolerance = max(0, int(contact_tolerance))
        self.relational_pair = None
        if self.relational:
            rng_state = torch.random.get_rng_state()
            self.relational_pair = nn.Sequential(
                nn.Linear(
                    int(hidden_dim) * 4 + 2,
                    int(relational_hidden),
                ),
                nn.SiLU(),
                nn.Linear(int(relational_hidden), 1),
            )
        if self.relational_pair is not None:
            nn.init.normal_(self.relational_pair[-1].weight, std=0.01)
            nn.init.zeros_(self.relational_pair[-1].bias)
            torch.random.set_rng_state(rng_state)
        self.contact_scale = (
            nn.Parameter(torch.tensor(0.5))
            if self.mask_aware
            else None
        )
        self.relational_scale = (
            nn.Parameter(torch.tensor(0.1))
            if self.relational
            else None
        )

    def forward(
        self,
        features: torch.Tensor,
        mask_logits: torch.Tensor | None = None,
        query_presence: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits = room_adjacency_logits(features, self.projection)
        if not self.mask_aware:
            return logits
        if mask_logits is None:
            raise ValueError("mask-aware adjacency requires mask logits")
        if mask_logits.shape[:2] != logits.shape[:2]:
            raise ValueError("mask queries must match adjacency queries")
        contact = mask_contact_logits(
            mask_logits,
            tolerance=self.contact_tolerance,
        )
        assert self.contact_scale is not None
        logits = logits + torch.tanh(self.contact_scale) * contact
        if (
            not self.relational
            or self.relational_pair is None
            or self.relational_scale is None
        ):
            return logits
        normalized = F.normalize(features.float(), dim=-1)
        left = normalized[:, :, None].expand(
            -1,
            -1,
            normalized.shape[1],
            -1,
        )
        right = normalized[:, None].expand_as(left)
        contact_probability = contact.float().sigmoid()
        if query_presence is None:
            pair_presence = torch.ones_like(contact_probability)
        else:
            if query_presence.shape != contact_probability.shape[:2]:
                raise ValueError("query presence must be [B,Q]")
            presence = query_presence.float().clamp(0.0, 1.0)
            pair_presence = (
                presence[:, :, None] * presence[:, None, :]
            )
        pair_features = torch.cat(
            [
                left,
                right,
                (left - right).abs(),
                left * right,
                contact_probability[..., None],
                pair_presence[..., None],
            ],
            dim=-1,
        )
        relational = self.relational_pair(pair_features).squeeze(-1)
        relational = 0.5 * (relational + relational.transpose(1, 2))
        return logits + torch.tanh(self.relational_scale) * relational


def room_graph_weights(
    adjacency_logits: torch.Tensor,
    query_presence: torch.Tensor | None = None,
) -> torch.Tensor:
    """Normalize room messages while suppressing absent query slots."""

    adjacency = adjacency_logits.sigmoid()
    diagonal = torch.eye(
        adjacency.shape[-1],
        device=adjacency.device,
        dtype=torch.bool,
    )[None]
    adjacency = adjacency.masked_fill(diagonal, 0.0)
    if query_presence is not None:
        if query_presence.shape != adjacency.shape[:2]:
            raise ValueError("query presence must be [B,Q]")
        pair_weight = (
            query_presence[..., None] * query_presence[..., None, :]
        )
        adjacency = adjacency * pair_weight.to(adjacency.dtype)
    return adjacency / adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)


class RoomGraphConditioner(nn.Module):
    """Propagate room-adjacency messages into identity and mask predictions."""

    def __init__(
        self,
        hidden_dim: int,
        feature_channels: int,
        room_classes: int,
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim)
        self.query_update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.class_residual = nn.Linear(hidden_dim, int(room_classes))
        self.mask_residual = nn.Linear(hidden_dim, int(feature_channels))
        self.scale = nn.Parameter(torch.tensor(0.1))
        self.reset_graph_residual_heads()

    def reset_graph_residual_heads(self, *, std: float = 0.01) -> None:
        """Warm-start graph-to-output residuals without changing old checkpoints."""

        with torch.no_grad():
            for head in (self.class_residual, self.mask_residual):
                nn.init.normal_(head.weight, std=float(std))
                nn.init.zeros_(head.bias)

    def forward(
        self,
        query_features: torch.Tensor,
        adjacency_logits: torch.Tensor,
        spatial_features: torch.Tensor,
        query_presence: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        weights = room_graph_weights(
            adjacency_logits,
            query_presence,
        )
        neighbor = torch.bmm(weights, query_features)
        updated = query_features + torch.tanh(self.scale) * self.query_update(
            torch.cat([query_features, neighbor], dim=-1)
        )
        mask_bias = self.mask_residual(updated)
        mask_residual = torch.einsum(
            "bqd,bdhw->bqhw",
            mask_bias,
            spatial_features.float(),
        )
        return {
            "query_features": updated,
            "class_residual": self.class_residual(updated),
            "mask_residual": mask_residual,
            "graph_conditioning_scale": self.scale,
        }


__all__ = [
    "RoomAdjacencyHead",
    "RoomGraphConditioner",
    "mask_contact_logits",
    "room_adjacency_logits",
    "room_graph_weights",
]
