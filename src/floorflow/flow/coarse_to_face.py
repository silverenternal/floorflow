"""Stage-2 residual flow conditioned on labeled coarse regions and faces."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CoarseToFaceFlowHead(nn.Module):
    def __init__(
        self,
        feature_channels: int,
        target_channels: int,
        num_room_classes: int,
        *,
        hidden_dim: int = 64,
        max_faces: int = 32,
        token_dim: int = 0,
    ) -> None:
        super().__init__()
        self.max_faces = int(max_faces)
        self.semantic_projection = nn.Sequential(
            nn.Conv2d(num_room_classes, hidden_dim, 3, padding=1),
            nn.GroupNorm(max(1, hidden_dim // 16), hidden_dim),
            nn.SiLU(),
        )
        self.face_embedding = nn.Embedding(self.max_faces + 1, hidden_dim)
        self.input_projection = nn.Conv2d(
            feature_channels + target_channels + 1,
            hidden_dim,
            3,
            padding=1,
        )
        self.time_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
                    nn.GroupNorm(max(1, hidden_dim // 16), hidden_dim),
                    nn.SiLU(),
                )
                for _ in range(2)
            ]
        )
        self.output = nn.Conv2d(hidden_dim, target_channels, 3, padding=1)
        self.output_scale = nn.Parameter(torch.zeros(1))
        self.semantic_token_projection = (
            nn.Linear(int(token_dim), hidden_dim) if token_dim > 0 else None
        )
        self.semantic_attention = (
            nn.MultiheadAttention(hidden_dim, 4, batch_first=True)
            if token_dim > 0
            else None
        )
        self.semantic_gate = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        x_t: torch.Tensor,
        time: torch.Tensor,
        features: torch.Tensor,
        footprint: torch.Tensor,
        coarse_logits: torch.Tensor,
        face_ids: torch.Tensor | None = None,
        semantic_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        size = features.shape[-2:]
        semantic = F.interpolate(
            coarse_logits.softmax(dim=1),
            size=size,
            mode="bilinear",
            align_corners=False,
        )
        semantic = self.semantic_projection(semantic)
        if face_ids is None:
            face_features = torch.zeros_like(semantic)
        else:
            face_features = self.face_embedding(
                face_ids.clamp(0, self.max_faces).long()
            ).permute(0, 3, 1, 2)
            face_features = F.interpolate(
                face_features,
                size=size,
                mode="nearest",
            )
        footprint = footprint[:, None] if footprint.ndim == 3 else footprint
        hidden = self.input_projection(
            torch.cat(
                [
                    features,
                    x_t,
                    footprint.to(x_t.dtype),
                ],
                dim=1,
            )
        )
        hidden = hidden + semantic + face_features
        if semantic_tokens is not None and self.semantic_token_projection is not None:
            tokens = self.semantic_token_projection(semantic_tokens)
            query = F.adaptive_avg_pool2d(hidden, (16, 16)).flatten(2).transpose(1, 2)
            context = self.semantic_attention(query, tokens, tokens, need_weights=False)[0]
            context = context.transpose(1, 2).reshape(
                hidden.shape[0], hidden.shape[1], 16, 16
            )
            context = F.interpolate(context, size=size, mode="bilinear", align_corners=False)
            hidden = hidden + self.semantic_gate.to(hidden.dtype) * context
        frequencies = torch.arange(
            hidden.shape[1] // 2,
            device=time.device,
            dtype=time.dtype,
        )
        angles = time[:, None] * frequencies[None] * math.pi
        time_features = torch.cat([angles.sin(), angles.cos()], dim=1)
        time_features = self.time_projection(time_features)
        for block in self.blocks:
            hidden = hidden + block(
                hidden
                + time_features[:, :, None, None]
            )
        residual = self.output(hidden)
        return self.output_scale.to(residual.dtype) * residual
