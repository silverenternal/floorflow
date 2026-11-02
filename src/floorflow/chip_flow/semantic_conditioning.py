"""Optional semantic node and port feature conditioning."""

from __future__ import annotations

import torch
from torch import nn


class MaskedSemanticEncoder(nn.Module):
    """Zero-gated semantic feature encoder with a fail-closed mask."""

    def __init__(
        self,
        input_dim: int,
        hidden: int,
        *,
        width: int | None = None,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or hidden <= 0:
            raise ValueError("input_dim and hidden must be positive")
        self.input_dim = int(input_dim)
        self.hidden = int(hidden)
        bottleneck = int(width or max(16, hidden // 8))
        self.network = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, bottleneck),
            nn.SiLU(),
            nn.Linear(bottleneck, hidden),
        )
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        features: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if features.ndim < 2 or features.shape[-1] != self.input_dim:
            raise ValueError(
                f"semantic features must end in width {self.input_dim}"
            )
        if mask is None:
            mask = features.new_ones(features.shape[:-1])
        if mask.shape != features.shape[:-1]:
            raise ValueError("semantic mask shape mismatch")
        valid = mask.bool()
        safe_features = torch.where(
            valid[..., None],
            features.float(),
            features.new_zeros(()),
        )
        encoded = self.gate * self.network(safe_features)
        return torch.where(valid[..., None], encoded, encoded.new_zeros(()))


__all__ = ["MaskedSemanticEncoder"]
