"""Small straight-through vector quantizer for two-level floorplan codes."""

from __future__ import annotations

import torch
from torch import nn

from floorflow.objectives.triton_ops import triton_vq_quantize


class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size: int, embedding_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(int(codebook_size), int(embedding_dim))
        self.embedding.weight.data.uniform_(
            -1.0 / codebook_size,
            1.0 / codebook_size,
        )

    def forward(
        self,
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return triton_vq_quantize(values, self.embedding.weight)


__all__ = ["VectorQuantizer"]
