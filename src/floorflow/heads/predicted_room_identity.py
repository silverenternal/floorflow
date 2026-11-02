"""Dense residual room-identity head over predicted contour controls."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class RoomIdentityResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(max(1, channels // 16), channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(max(1, channels // 16), channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(inputs)))
        return inputs + self.conv2(F.silu(self.norm2(hidden)))


class PredictedRoomIdentityHead(nn.Module):
    """Predict dense room-class corrections without changing the base prior.

    The output convolution is zero-initialized, so a new checkpoint is an exact
    identity to the frozen segmenter prediction while its residual path still
    receives useful gradients on the first optimization step.
    """

    def __init__(
        self,
        in_channels: int,
        num_room_classes: int,
        hidden_channels: int = 96,
        blocks: int = 3,
        use_image: bool = False,
        image_channels: int = 3,
    ) -> None:
        super().__init__()
        self.input = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(max(1, hidden_channels // 16), hidden_channels),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(
            *[
                RoomIdentityResidualBlock(hidden_channels)
                for _ in range(max(1, int(blocks)))
            ]
        )
        self.output = nn.Conv2d(hidden_channels, num_room_classes, 1)
        self.use_image = bool(use_image)
        self.image_input = (
            nn.Sequential(
                nn.Conv2d(image_channels, hidden_channels, 3, padding=1),
                nn.GroupNorm(max(1, hidden_channels // 16), hidden_channels),
                nn.SiLU(),
            )
            if self.use_image
            else None
        )
        self.image_blocks = (
            nn.Sequential(
                *[
                    RoomIdentityResidualBlock(hidden_channels)
                    for _ in range(max(1, int(blocks)))
                ]
            )
            if self.use_image
            else None
        )
        self.image_output = (
            nn.Conv2d(hidden_channels, hidden_channels, 1)
            if self.use_image
            else None
        )
        self.residual_scale = nn.Parameter(torch.ones(1))
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        if self.image_output is not None:
            nn.init.zeros_(self.image_output.weight)
            nn.init.zeros_(self.image_output.bias)

    def forward(
        self,
        features: torch.Tensor,
        base_logits: torch.Tensor,
        image: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if features.shape[-2:] != base_logits.shape[-2:]:
            raise ValueError("features and base logits must share a resolution")
        hidden = self.blocks(self.input(features))
        if self.use_image:
            if image is None:
                raise ValueError("image input is required")
            if image.shape[-2:] != features.shape[-2:]:
                image = F.interpolate(
                    image,
                    size=features.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            hidden = hidden + self.image_output(
                self.image_blocks(self.image_input(image))
            )
        residual = self.output(hidden)
        corrected = (
            base_logits
            + self.residual_scale.to(residual.dtype) * residual
        )
        return {
            "room_logits": corrected,
            "residual_logits": residual,
            "residual_scale": self.residual_scale,
        }


def room_identity_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    footprint: torch.Tensor,
    room_ids: tuple[int, ...],
    *,
    class_weights: torch.Tensor | None = None,
    dice_weight: float = 0.5,
    boundary_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Class-balanced dense CE, Dice, and boundary-aware room loss."""

    local = torch.full_like(labels, -1)
    for index, class_id in enumerate(room_ids):
        local[labels == int(class_id)] = index
    valid = (local >= 0) & footprint.bool()
    ce = F.cross_entropy(
        logits,
        local.clamp_min(0),
        weight=class_weights,
        reduction="none",
    )
    boundary_target = valid & ~(
        (
            -F.max_pool2d(
                -valid.float()[:, None],
                5,
                stride=1,
                padding=2,
            )
        )[:, 0].bool()
    )
    weights = valid.float() * (
        1.0 + float(boundary_weight) * boundary_target.float()
    )
    ce_loss = (ce * weights).sum() / weights.sum().clamp_min(1.0)
    probability = logits.softmax(dim=1)
    one_hot = F.one_hot(
        local.clamp_min(0),
        len(room_ids),
    ).permute(0, 3, 1, 2).float()
    valid_float = valid[:, None].float()
    intersection = (probability * one_hot).sum(dim=(0, 2, 3))
    denominator = (
        (probability * valid_float).sum(dim=(0, 2, 3))
        + (one_hot * valid_float).sum(dim=(0, 2, 3))
    )
    present = (one_hot * valid_float).sum(dim=(0, 2, 3)) > 0
    dice_values = torch.nan_to_num(
        (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)).clamp(
            0.0,
            2.0,
        ),
        nan=0.0,
        posinf=2.0,
        neginf=0.0,
    )
    dice = (dice_values * present).sum() / present.sum().clamp_min(1)
    total = ce_loss + float(dice_weight) * dice
    return total, {
        "room_identity_ce": ce_loss.detach(),
        "room_identity_dice": dice.detach(),
        "room_identity_boundary": boundary_target.float().mean().detach(),
    }


__all__ = [
    "PredictedRoomIdentityHead",
    "RoomIdentityResidualBlock",
    "room_identity_loss",
]
