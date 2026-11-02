"""Reliability calibration for hierarchical parent masks."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn


class ParentReliabilityGate(nn.Module):
    """Learn when sparse parent support should be normalized to full occupancy."""

    def __init__(self, groups: Sequence[str]) -> None:
        super().__init__()
        self.blend = nn.ParameterDict(
            {name: nn.Parameter(torch.zeros(1)) for name in groups}
        )

    def forward(
        self,
        name: str,
        parent: torch.Tensor,
        footprint: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if footprint.ndim == 3:
            footprint = footprint[:, None]
        inside = footprint.float()
        if inside.shape[-2:] != parent.shape[-2:]:
            inside = F.interpolate(
                inside,
                size=parent.shape[-2:],
                mode="nearest",
            )
        coverage = (parent * inside).sum(dim=(1, 2, 3), keepdim=True) / inside.sum(
            dim=(1, 2, 3),
            keepdim=True,
        ).clamp_min(1.0)
        normalized = (parent / coverage.clamp_min(1.0e-4)).clamp(0.0, 1.0)
        blend = torch.tanh(self.blend[name]).to(parent.dtype)
        calibrated = (1.0 - blend) * parent + blend * normalized
        calibrated = calibrated * inside
        effect = (calibrated - parent).abs() * inside
        return calibrated, {
            "coverage": coverage.mean().detach(),
            "blend": blend.detach(),
            "effect": (
                effect.sum(dim=(1, 2, 3), keepdim=True)
                / inside.sum(dim=(1, 2, 3), keepdim=True).clamp_min(1.0)
            ).mean().detach(),
        }


__all__ = ["ParentReliabilityGate"]
