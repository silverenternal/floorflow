"""Second-stage boundary refinement for a coarse room-program layout."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class _Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(max(1, out_channels // 16), out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(max(1, out_channels // 16), out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CoarseLayoutRefiner(nn.Module):
    """A small U-Net that turns a coarse layout prior into room logits."""

    def __init__(
        self,
        num_room_classes: int,
        base_channels: int = 32,
        *,
        residual: bool = False,
        masked_input: bool = False,
        zero_masked_input: bool = True,
    ) -> None:
        super().__init__()
        base = int(base_channels)
        self.residual = bool(residual)
        self.masked_input = bool(masked_input)
        self.zero_masked_input = bool(zero_masked_input)
        mask_channels = 1 if self.masked_input else 0
        self.enc1 = _Block(num_room_classes + 1 + mask_channels, base)
        self.enc2 = _Block(base, base * 2)
        self.enc3 = _Block(base * 2, base * 4)
        self.dec2 = _Block(base * 6, base * 2)
        self.dec1 = _Block(base * 3, base)
        self.out = nn.Conv2d(base, num_room_classes, 1)
        if self.residual:
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

    def forward(
        self,
        coarse_logits: torch.Tensor,
        footprint: torch.Tensor,
        validity_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if footprint.ndim == 3:
            footprint = footprint[:, None]
        if self.masked_input:
            if validity_mask is None:
                raise ValueError("validity_mask is required for masked_input")
            if validity_mask.ndim == 3:
                validity_mask = validity_mask[:, None]
            validity_mask = F.interpolate(
                validity_mask.float(),
                size=coarse_logits.shape[-2:],
                mode="nearest",
            )
            if self.zero_masked_input:
                coarse_logits = coarse_logits * validity_mask
        coarse_logits = F.interpolate(
            coarse_logits,
            size=footprint.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        if self.masked_input:
            validity_mask = F.interpolate(
                validity_mask,
                size=footprint.shape[-2:],
                mode="nearest",
            )
        channels = [coarse_logits, footprint.float()]
        if self.masked_input:
            channels.append(validity_mask)
        x = torch.cat(channels, dim=1)
        skip1 = self.enc1(x)
        skip2 = self.enc2(F.avg_pool2d(skip1, 2))
        low = self.enc3(F.avg_pool2d(skip2, 2))
        x = F.interpolate(low, size=skip2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec2(torch.cat([x, skip2], dim=1))
        x = F.interpolate(x, size=skip1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec1(torch.cat([x, skip1], dim=1))
        refined = self.out(x)
        return coarse_logits + refined if self.residual else refined


__all__ = ["CoarseLayoutRefiner"]
