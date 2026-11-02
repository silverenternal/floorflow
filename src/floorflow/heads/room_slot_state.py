"""Joint room-slot state for identity, presence, and cardinality."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from floorflow.heads.room_mask_decoder import query_existence_from_class_logits
from floorflow.heads.room_owner_readout import (
    exclusive_query_owner_probability,
)


def poisson_binomial_log_probability(
    presence_logits: torch.Tensor,
) -> torch.Tensor:
    """Derive a count distribution from one Bernoulli state per slot."""

    if presence_logits.ndim != 2:
        raise ValueError("presence_logits must be [B,Q]")
    probability = presence_logits.sigmoid()
    log_positive = probability.clamp_min(1.0e-8).log()
    log_negative = (1.0 - probability).clamp_min(1.0e-8).log()
    distribution = presence_logits.new_zeros(
        (presence_logits.shape[0], 1)
    )
    for slot in range(presence_logits.shape[1]):
        absent = torch.cat(
            [
                distribution + log_negative[:, slot, None],
                distribution.new_full((distribution.shape[0], 1), -1.0e30),
            ],
            dim=1,
        )
        present = torch.cat(
            [
                distribution.new_full((distribution.shape[0], 1), -1.0e30),
                distribution + log_positive[:, slot, None],
            ],
            dim=1,
        )
        distribution = torch.logaddexp(absent, present)
    return distribution


class RoomSlotStateHead(nn.Module):
    """Refine slot class/presence from shared mask-conditioned slot states."""

    def __init__(
        self,
        query_dim: int,
        feature_channels: int,
        num_room_classes: int,
        num_queries: int,
        *,
        hidden_dim: int = 64,
        heads: int = 4,
        pool_size: int = 48,
    ) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.num_room_classes = int(num_room_classes)
        self.num_queries = int(num_queries)
        self.pool_size = max(1, int(pool_size))
        hidden = int(hidden_dim)
        self.feature_projection = nn.Conv2d(
            int(feature_channels),
            hidden,
            1,
        )
        self.query_projection = nn.Linear(int(query_dim), hidden)
        self.descriptor_projection = nn.Sequential(
            nn.Linear(7, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.self_attention = nn.MultiheadAttention(
            hidden,
            int(heads),
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden)
        self.presence_residual = nn.Linear(hidden, 1)
        self.class_residual = nn.Linear(
            hidden,
            self.num_room_classes + 1,
        )
        self.mask_state_projection = nn.Linear(hidden, 1)
        for module in (
            self.presence_residual,
            self.class_residual,
            self.mask_state_projection,
        ):
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        features: torch.Tensor,
        query_features: torch.Tensor,
        mask_logits: torch.Tensor,
        class_logits: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if features.ndim != 4:
            raise ValueError("features must be [B,C,H,W]")
        if query_features.ndim != 3:
            raise ValueError("query_features must be [B,Q,D]")
        if mask_logits.ndim != 4 or class_logits.ndim != 3:
            raise ValueError("mask/class logits must be [B,Q,H,W]/[B,Q,C+1]")
        if mask_logits.shape[1] != query_features.shape[1]:
            raise ValueError("mask and query slot counts must match")
        if class_logits.shape[:2] != query_features.shape[:2]:
            raise ValueError("class and query slots must match")
        if mask_logits.shape[1] != self.num_queries:
            raise ValueError("mask slot count does not match the state head")
        size = tuple(
            min(self.pool_size, dimension)
            for dimension in features.shape[-2:]
        )
        full_projected = self.feature_projection(features.float())
        projected = F.adaptive_avg_pool2d(full_projected, size)
        probability = F.adaptive_avg_pool2d(
            mask_logits.float().sigmoid(),
            size,
        )
        normalized = probability / probability.sum(
            dim=(2, 3),
            keepdim=True,
        ).clamp_min(1.0e-6)
        pooled = torch.einsum(
            "bqhw,bchw->bqc",
            normalized,
            projected,
        )
        mass = probability.mean(dim=(2, 3))
        peak = probability.amax(dim=(2, 3))
        area = (probability >= 0.5).float().mean(dim=(2, 3))
        coordinates = torch.linspace(
            0.0,
            1.0,
            size[0],
            device=features.device,
            dtype=features.dtype,
        )
        grid_y, grid_x = torch.meshgrid(
            coordinates,
            coordinates,
            indexing="ij",
        )
        center_x = (
            probability * grid_x[None, None]
        ).sum(dim=(2, 3)) / probability.sum(
            dim=(2, 3)
        ).clamp_min(1.0e-6)
        center_y = (
            probability * grid_y[None, None]
        ).sum(dim=(2, 3)) / probability.sum(
            dim=(2, 3)
        ).clamp_min(1.0e-6)
        adjacent = torch.bmm(
            normalized.flatten(2),
            normalized.flatten(2).transpose(1, 2),
        )
        eye = torch.eye(
            self.num_queries,
            device=features.device,
            dtype=torch.bool,
        )[None]
        overlap = adjacent.masked_fill(eye, 0.0).amax(dim=-1)
        class_probability = class_logits.float().softmax(dim=-1)
        class_entropy = -(
            class_probability.clamp_min(1.0e-8)
            * class_probability.clamp_min(1.0e-8).log()
        ).sum(dim=-1)
        descriptors = torch.stack(
            [
                mass,
                peak,
                area,
                center_x,
                center_y,
                overlap,
                class_entropy,
            ],
            dim=-1,
        )
        slot_state = (
            self.query_projection(query_features.float())
            + self.descriptor_projection(descriptors)
            + pooled
        )
        attended = self.self_attention(
            slot_state,
            slot_state,
            slot_state,
            need_weights=False,
        )[0]
        slot_state = self.norm(slot_state + attended)
        base_presence = query_existence_from_class_logits(class_logits)
        presence = (
            base_presence
            + self.presence_residual(slot_state).squeeze(-1)
        )
        class_residual = self.class_residual(slot_state)
        classes = class_logits + class_residual
        mask_residual = torch.einsum(
            "bq,bdhw->bqhw",
            self.mask_state_projection(slot_state).squeeze(-1),
            full_projected,
        )
        refined_masks = F.interpolate(
            mask_logits.float(),
            size=features.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ) + mask_residual
        owner_probability = exclusive_query_owner_probability(refined_masks)
        count_log_probability = poisson_binomial_log_probability(presence)
        return {
            "class_logits": classes,
            "query_existence_logits": presence,
            "mask_logits": refined_masks,
            "room_count_logits": count_log_probability,
            "slot_state_features": slot_state,
            "slot_adjacency": adjacent,
            "slot_class_residual": class_residual,
            "slot_owner_probability": owner_probability,
            "slot_count_probability": count_log_probability.exp(),
            "slot_state_diagnostics": {
                "presence_mean": presence.sigmoid().mean(),
                "expected_count": presence.sigmoid().sum(dim=-1).mean(),
                "duplicate_overlap": overlap.mean(),
                "mask_residual_norm": mask_residual.float().square().sum(
                    dim=(1, 2, 3)
                ).sqrt().mean(),
            },
        }


__all__ = [
    "RoomSlotStateHead",
    "poisson_binomial_log_probability",
]
