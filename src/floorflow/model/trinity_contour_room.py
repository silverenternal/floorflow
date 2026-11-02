"""Contour encoder and Trinity denoiser for room-latent flow matching."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from trinity.models.backbone import DenoiserCond, SetTransformerDenoiser
from trinity.models.presets import DenoiserArchConfig

from floorflow.model.mixture_topk_attention import MixtureTopKAttention


class ContourRoomTokenEncoder(nn.Module):
    """Encode contour pixels into a fixed-size set of spatial room tokens."""

    def __init__(
        self,
        hidden: int = 96,
        room_tokens: int = 32,
        heads: int = 4,
        grid: int = 16,
    ) -> None:
        super().__init__()
        self.grid = int(grid)
        self.hidden = int(hidden)
        self.room_tokens = int(room_tokens)
        self.trunk = nn.Sequential(
            nn.Conv2d(1, hidden // 2, 3, padding=1),
            nn.GroupNorm(max(1, hidden // 16), hidden // 2),
            nn.SiLU(),
            nn.Conv2d(hidden // 2, hidden, 3, padding=1),
            nn.GroupNorm(max(1, hidden // 16), hidden),
            nn.SiLU(),
        )
        self.queries = nn.Parameter(torch.randn(room_tokens, hidden) * 0.02)
        self.attention = MixtureTopKAttention(
            hidden,
            num_heads=heads,
            router_count=min(16, room_tokens),
            top_k=min(8, grid * grid),
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(self, footprint: torch.Tensor) -> tuple[torch.Tensor, dict]:
        if footprint.ndim == 3:
            footprint = footprint[:, None]
        hidden = F.adaptive_avg_pool2d(
            self.trunk(footprint.float()),
            (self.grid, self.grid),
        )
        memory = hidden.flatten(2).transpose(1, 2)
        queries = self.queries[None].expand(footprint.shape[0], -1, -1)
        tokens, diagnostics = self.attention(queries, memory)
        return self.norm(tokens), diagnostics


class ContourTrinityRoomModel(nn.Module):
    """Predict Trinity room latents from a contour-only token conditioner."""

    def __init__(
        self,
        *,
        hidden: int = 96,
        depth: int = 4,
        heads: int = 4,
        room_tokens: int = 32,
        grid: int = 16,
        num_room_classes: int = 24,
    ) -> None:
        super().__init__()
        self.room_tokens = int(room_tokens)
        self.encoder = ContourRoomTokenEncoder(
            hidden=hidden,
            room_tokens=room_tokens,
            heads=heads,
            grid=grid,
        )
        config = DenoiserArchConfig(
            latent_dim=3,
            feature_dim=hidden,
            hidden=hidden,
            depth=depth,
            heads=heads,
            graph_pe_dim=0,
            graph_bias=False,
            output_kind="target",
        )
        self.denoiser = SetTransformerDenoiser(config)
        self.token_head = nn.Linear(hidden, 1)
        self.class_head = nn.Linear(hidden, int(num_room_classes))
        self.area_head = nn.Linear(hidden, 1)

    def forward_with_aux(
        self,
        z_t: torch.Tensor,
        time: torch.Tensor,
        footprint: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        features, routing = self.encoder(footprint)
        condition = DenoiserCond(
            features=features,
            adjacency=None,
            key_pad_mask=None,
            graph_pe=None,
        )
        prediction = self.denoiser(z_t, time, condition)
        return prediction, {
            "token_logits": self.token_head(features).squeeze(-1),
            "class_logits": self.class_head(features),
            "log_area": self.area_head(features).squeeze(-1),
            "routing_entropy": routing["routing_entropy"],
            "effective_expert_fraction": routing["effective_expert_fraction"],
        }

    def forward(
        self,
        z_t: torch.Tensor,
        time: torch.Tensor,
        footprint: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_with_aux(z_t, time, footprint)[0]


class FeatureRoomTokenEncoder(nn.Module):
    """Adapt pretrained condition features into a set of Trinity room tokens."""

    def __init__(
        self,
        in_channels: int,
        hidden: int = 96,
        room_tokens: int = 32,
        heads: int = 4,
        grid: int = 16,
    ) -> None:
        super().__init__()
        self.grid = int(grid)
        self.hidden = int(hidden)
        self.room_tokens = int(room_tokens)
        self.project = nn.Sequential(
            nn.Conv2d(int(in_channels), hidden, 1),
            nn.GroupNorm(max(1, hidden // 16), hidden),
            nn.SiLU(),
        )
        self.queries = nn.Parameter(torch.randn(room_tokens, hidden) * 0.02)
        self.attention = MixtureTopKAttention(
            hidden,
            num_heads=heads,
            router_count=min(16, room_tokens),
            top_k=min(8, grid * grid),
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, dict]:
        hidden = F.adaptive_avg_pool2d(
            self.project(features.float()),
            (self.grid, self.grid),
        )
        memory = hidden.flatten(2).transpose(1, 2)
        queries = self.queries[None].expand(features.shape[0], -1, -1)
        tokens, diagnostics = self.attention(queries, memory)
        return self.norm(tokens), diagnostics


class FeatureTrinityRoomModel(nn.Module):
    """Trinity room denoiser conditioned on frozen contour features."""

    def __init__(
        self,
        in_channels: int,
        *,
        hidden: int = 96,
        depth: int = 4,
        heads: int = 4,
        room_tokens: int = 32,
        grid: int = 16,
        num_room_classes: int = 24,
    ) -> None:
        super().__init__()
        self.room_tokens = int(room_tokens)
        self.encoder = FeatureRoomTokenEncoder(
            in_channels,
            hidden=hidden,
            room_tokens=room_tokens,
            heads=heads,
            grid=grid,
        )
        config = DenoiserArchConfig(
            latent_dim=3,
            feature_dim=hidden,
            hidden=hidden,
            depth=depth,
            heads=heads,
            graph_pe_dim=0,
            graph_bias=False,
            output_kind="target",
        )
        self.denoiser = SetTransformerDenoiser(config)
        self.token_head = nn.Linear(hidden, 1)
        self.class_head = nn.Linear(hidden, int(num_room_classes))
        self.area_head = nn.Linear(hidden, 1)

    def forward_with_aux(
        self,
        z_t: torch.Tensor,
        time: torch.Tensor,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        tokens, routing = self.encoder(features)
        prediction = self.denoiser(
            z_t,
            time,
            DenoiserCond(features=tokens),
        )
        return prediction, {
            "token_logits": self.token_head(tokens).squeeze(-1),
            "class_logits": self.class_head(tokens),
            "log_area": self.area_head(tokens).squeeze(-1),
            "routing_entropy": routing["routing_entropy"],
            "effective_expert_fraction": routing["effective_expert_fraction"],
        }

    def forward(
        self,
        z_t: torch.Tensor,
        time: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_with_aux(z_t, time, features)[0]


__all__ = [
    "ContourRoomTokenEncoder",
    "ContourTrinityRoomModel",
    "FeatureRoomTokenEncoder",
    "FeatureTrinityRoomModel",
]
