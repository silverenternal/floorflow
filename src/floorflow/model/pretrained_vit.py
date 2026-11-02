"""Frozen timm ViT patch-token adapter for pixel distribution heads."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrozenViTPatchFeatures(nn.Module):
    """Expose a pretrained ViT's patch tokens as a spatial feature map."""

    def __init__(
        self,
        backbone_name: str = "vit_small_patch32_384.augreg_in21k_ft_in1k",
        image_size: int = 256,
    ) -> None:
        super().__init__()
        import timm

        self.image_size = int(image_size)
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            num_classes=0,
            img_size=self.image_size,
        )
        self.backbone.eval().requires_grad_(False)
        self.channels = int(self.backbone.embed_dim)
        patch_size = self.backbone.patch_embed.patch_size
        patch = patch_size[0] if isinstance(patch_size, tuple) else patch_size
        self.grid = max(1, self.image_size // int(patch))

    @torch.no_grad()
    def forward(self, footprint: torch.Tensor) -> torch.Tensor:
        if footprint.ndim == 3:
            footprint = footprint[:, None]
        value = footprint.float()
        if value.shape[-2:] != (self.image_size, self.image_size):
            value = F.interpolate(
                value,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        value = value.expand(-1, 3, -1, -1)
        value = (value - 0.5) / 0.25
        tokens = self.backbone.forward_features(value)
        prefix = int(self.backbone.num_prefix_tokens)
        patch_tokens = tokens[:, prefix:]
        side = int(round(math.sqrt(patch_tokens.shape[1])))
        if side * side != patch_tokens.shape[1]:
            raise RuntimeError("ViT patch token grid is not square")
        return (
            patch_tokens.transpose(1, 2)
            .reshape(value.shape[0], self.channels, side, side)
            .contiguous()
        )


__all__ = ["FrozenViTPatchFeatures"]
