"""Align the predicted block envelope with the model's L0 forecast."""

from __future__ import annotations

import torch
from torch import nn

from floorflow.chip_flow.model import decode_boxes_batch


class GlobalBBoxConsensus(nn.Module):
    """Apply a learned affine envelope correction while preserving layout."""

    def __init__(self) -> None:
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        block_prediction: torch.Tensor,
        global_prediction: torch.Tensor,
        areas: torch.Tensor,
        constraints: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, count, _ = block_prediction.shape
        if global_prediction.shape != (batch, 4):
            raise ValueError("global_prediction must have shape (batch, 4)")
        if areas.shape != (batch, count):
            raise ValueError("areas must have shape (batch, n)")
        boxes = decode_boxes_batch(block_prediction.float(), areas.float())
        scale = areas.float().sum(dim=1).sqrt().clamp_min(1.0e-6)
        current_center = torch.stack(
            [
                (boxes[..., 0].amin(dim=1) + (boxes[..., 0] + boxes[..., 2]).amax(dim=1))
                * 0.5,
                (boxes[..., 1].amin(dim=1) + (boxes[..., 1] + boxes[..., 3]).amax(dim=1))
                * 0.5,
            ],
            dim=-1,
        ) / scale[:, None]
        current_size = torch.stack(
            [
                (boxes[..., 0] + boxes[..., 2]).amax(dim=1)
                - boxes[..., 0].amin(dim=1),
                (boxes[..., 1] + boxes[..., 3]).amax(dim=1)
                - boxes[..., 1].amin(dim=1),
            ],
            dim=-1,
        ) / scale[:, None]
        target_center = global_prediction[:, :2].float()
        target_size = global_prediction[:, 2:].float()
        ratio = (
            target_size / current_size.clamp_min(1.0e-6)
        ).clamp(0.25, 4.0)
        corrected_center = target_center[:, None] + (
            block_prediction[..., :2].float() - current_center[:, None]
        ) * ratio[:, None]
        strength = torch.tanh(self.gate)
        centers = torch.lerp(
            block_prediction[..., :2].float(),
            corrected_center,
            strength,
        )
        if constraints is not None:
            if constraints.shape != (batch, count, 5):
                raise ValueError("constraints must have shape [B,N,5]")
            preplaced = constraints[..., 1].ne(0)
            centers = torch.where(
                preplaced[..., None],
                block_prediction[..., :2].float(),
                centers,
            )
        output = torch.cat(
            [centers, block_prediction[..., 2:].float()],
            dim=-1,
        ).to(block_prediction.dtype)
        residuals = (output[..., :2] - block_prediction[..., :2]).abs()
        return output, residuals.mean(dim=(1, 2))


__all__ = ["GlobalBBoxConsensus"]
