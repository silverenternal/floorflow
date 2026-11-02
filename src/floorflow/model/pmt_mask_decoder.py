"""PMT-style joint mask decoder adapted to dense floorplan features."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from floorflow.heads.masked_channel_distribution import _position_embedding
from floorflow.heads.pixel_distribution import PixelDistributionHead
from floorflow.model.multiscale_pixel_decoder import ScaleBlockLite


class _JointAttentionBlock(nn.Module):
    """Joint class/patch transformer block used by the PMT decoder."""

    def __init__(
        self,
        dim: int,
        heads: int,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.heads = int(heads)
        self.head_dim = dim // self.heads
        self.dropout = float(dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
        )
        self.scale1 = nn.Parameter(torch.ones(dim))
        self.scale2 = nn.Parameter(torch.ones(dim))

    def forward(
        self,
        tokens: torch.Tensor,
        allowed: torch.Tensor | None,
    ) -> torch.Tensor:
        batch, count, dim = tokens.shape
        qkv = self.qkv(self.norm1(tokens)).reshape(
            batch,
            count,
            3,
            self.heads,
            self.head_dim,
        )
        query, key, value = qkv.permute(2, 0, 3, 1, 4)
        attention_mask = None
        if allowed is not None:
            attention_mask = allowed[:, None].expand(
                -1,
                self.heads,
                -1,
                -1,
            )
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(batch, count, dim)
        tokens = tokens + self.scale1 * self.proj(attended)
        return tokens + self.scale2 * self.mlp(self.norm2(tokens))


class PMTMaskDecoderAdapter(nn.Module):
    """Decode class masks while class queries and patch tokens co-evolve.

    The adapter follows PMT's Plain Mask Decoder: one token sequence contains
    class queries and feature patches, each block updates both, and class
    queries progressively attend only to patches currently inside their mask.
    It intentionally operates on local dense features so no pretrained token
    vocabulary or Python-version-specific PMT training stack is required.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hidden_channels: int = 96,
        heads: int = 4,
        blocks: int = 2,
        grid: int = 16,
        mask_dropout: float = 0.25,
        upscale_steps: int = 2,
        pixel_refiner: bool = False,
    ) -> None:
        super().__init__()
        if hidden_channels % heads:
            raise ValueError("hidden_channels must be divisible by heads")
        self.num_classes = int(num_classes)
        self.grid = int(grid)
        self.upscale_steps = int(upscale_steps)
        self.mask_dropout = float(mask_dropout)
        if not 0.0 <= self.mask_dropout <= 1.0:
            raise ValueError("mask_dropout must be in [0, 1]")
        if self.upscale_steps < 0:
            raise ValueError("upscale_steps must be non-negative")
        self.feature_projection = nn.Sequential(
            nn.Conv2d(int(in_channels), hidden_channels, 1),
            nn.GroupNorm(max(1, hidden_channels // 16), hidden_channels),
            nn.SiLU(),
        )
        self.queries = nn.Parameter(
            torch.randn(self.num_classes, hidden_channels) * 0.02
        )
        self.blocks = nn.ModuleList(
            [
                _JointAttentionBlock(hidden_channels, heads)
                for _ in range(int(blocks))
            ]
        )
        self.decoder_norm = nn.LayerNorm(hidden_channels)
        self.mask_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.mask_upscale = nn.ModuleList(
            [ScaleBlockLite(hidden_channels) for _ in range(self.upscale_steps)]
        )
        self.pixel_refiner = (
            PixelDistributionHead(
                in_channels,
                self.num_classes,
                hidden_channels=hidden_channels,
            )
            if pixel_refiner
            else None
        )
        if self.pixel_refiner is not None:
            nn.init.zeros_(self.pixel_refiner.mix[-1].weight)
            nn.init.zeros_(self.pixel_refiner.mix[-1].bias)
        self.register_buffer("attn_mask_probs", torch.ones(int(blocks)))

    def _predict(
        self,
        class_tokens: torch.Tensor,
        patch_tokens: torch.Tensor,
    ) -> torch.Tensor:
        class_embedding = self.mask_head(self.decoder_norm(class_tokens))
        patch_embedding = self.decoder_norm(patch_tokens)
        logits = torch.einsum(
            "bqd,bnd->bqn",
            class_embedding,
            patch_embedding,
        ) / math.sqrt(class_embedding.shape[-1])
        return logits.reshape(
            class_tokens.shape[0],
            self.num_classes,
            self.grid,
            self.grid,
        )

    def _predict_high_resolution(
        self,
        class_tokens: torch.Tensor,
        patch_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if not self.mask_upscale:
            return self._predict(class_tokens, patch_tokens)
        patch_map = patch_tokens.transpose(1, 2).reshape(
            patch_tokens.shape[0],
            patch_tokens.shape[-1],
            self.grid,
            self.grid,
        )
        for upscale in self.mask_upscale:
            patch_map = upscale(patch_map)
        class_embedding = self.mask_head(self.decoder_norm(class_tokens))
        patch_embedding = self.decoder_norm(
            patch_map.flatten(2).transpose(1, 2)
        )
        logits = torch.einsum(
            "bqd,bnd->bqn",
            class_embedding,
            patch_embedding,
        ) / math.sqrt(class_embedding.shape[-1])
        return logits.reshape(
            class_tokens.shape[0],
            self.num_classes,
            patch_map.shape[-2],
            patch_map.shape[-1],
        )

    def _attention_allowance(
        self,
        class_count: int,
        patch_count: int,
        mask_logits: torch.Tensor | None,
        block_index: int,
        device: torch.device,
    ) -> torch.Tensor:
        batch = 1 if mask_logits is None else mask_logits.shape[0]
        total = class_count + patch_count
        allowed = torch.ones(
            batch,
            total,
            total,
            dtype=torch.bool,
            device=device,
        )
        if mask_logits is None:
            return allowed
        flat_mask = mask_logits.flatten(2) > 0
        allowed[:, :class_count, class_count:] = flat_mask
        prob = float(self.attn_mask_probs[block_index].detach())
        if self.training and prob < 1.0:
            drop = torch.rand(
                mask_logits.shape[0],
                class_count,
                device=device,
            ) > prob
            if drop.any():
                allowed[:, :class_count, class_count:][drop] = True
        class_to_patch = allowed[:, :class_count, class_count:]
        empty = ~class_to_patch.any(dim=-1)
        class_to_patch.masked_fill_(empty.unsqueeze(-1), True)
        return allowed

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        projection = self.feature_projection(features.float())
        patches = F.adaptive_avg_pool2d(
            projection,
            (self.grid, self.grid),
        ).flatten(2).transpose(1, 2)
        patches = patches + _position_embedding(
            self.grid,
            self.grid,
            patches.shape[-1],
            patches.device,
        ).to(patches.dtype)
        queries = self.queries[None].expand(features.shape[0], -1, -1)
        mask_logits = None
        auxiliary_masks: list[torch.Tensor] = []
        for index, block in enumerate(self.blocks):
            if mask_logits is not None:
                auxiliary_masks.append(mask_logits)
            allowed = self._attention_allowance(
                queries.shape[1],
                patches.shape[1],
                mask_logits,
                index,
                features.device,
            )
            joint = torch.cat([queries, patches], dim=1)
            joint = block(joint, allowed)
            queries, patches = joint.split(
                [queries.shape[1], patches.shape[1]],
                dim=1,
            )
            mask_logits = self._predict(queries, patches)
        high_resolution_logits = self._predict_high_resolution(queries, patches)
        full_logits = F.interpolate(
            high_resolution_logits,
            size=features.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        if self.pixel_refiner is not None:
            full_logits = full_logits + self.pixel_refiner(features.float())["logits"]
        probability = full_logits.softmax(dim=1)
        entropy = -(
            probability.clamp_min(1.0e-8)
            * probability.clamp_min(1.0e-8).log()
        ).sum(dim=1, keepdim=True)
        reliability = 1.0 - entropy / max(
            1.0e-6,
            math.log(float(self.num_classes)),
        )
        auxiliary_masks.append(mask_logits)
        return {
            "logits": full_logits,
            "probability": probability,
            "room_logits": full_logits[:, 1:],
            "mask_logits": high_resolution_logits,
            "aux_mask_logits": auxiliary_masks,
            "reliability": reliability,
            "channel_tokens": self.decoder_norm(queries),
        }


__all__ = ["PMTMaskDecoderAdapter"]
