"""Block-token flow matching with exact-area decode and optional sparse attention."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

RHO_CLAMP = 4.0


def encode_latent(boxes: torch.Tensor, areas: torch.Tensor) -> torch.Tensor:
    """Boxes ``[N,4]`` -> latent ``[N,3]`` = (cx/s, cy/s, log(w/h))."""

    scale = areas.sum().sqrt().clamp_min(1.0e-6)
    center = (boxes[:, :2] + boxes[:, 2:] * 0.5) / scale
    rho = torch.log(boxes[:, 2].clamp_min(1.0e-6) / boxes[:, 3].clamp_min(1.0e-6))
    return torch.cat([center, rho[:, None].clamp(-RHO_CLAMP, RHO_CLAMP)], dim=1)


def encode_latents(boxes: torch.Tensor, areas: torch.Tensor) -> torch.Tensor:
    """Batched version of :func:`encode_latent` for ``[B, N, ...]`` tensors."""

    scale = areas.sum(dim=1).sqrt().clamp_min(1.0e-6)
    center = (boxes[..., :2] + boxes[..., 2:] * 0.5) / scale[:, None, None]
    rho = torch.log(
        boxes[..., 2].clamp_min(1.0e-6) / boxes[..., 3].clamp_min(1.0e-6)
    )
    return torch.cat(
        [center, rho[..., None].clamp(-RHO_CLAMP, RHO_CLAMP)],
        dim=-1,
    )


def decode_boxes(latent: torch.Tensor, areas: torch.Tensor) -> torch.Tensor:
    """Latent ``[N,3]`` -> boxes ``[N,4]`` with exact block areas."""

    scale = areas.sum().sqrt().clamp_min(1.0e-6)
    rho = latent[:, 2].clamp(-RHO_CLAMP, RHO_CLAMP)
    area = areas.clamp_min(1.0e-6)
    width = (area * torch.exp(rho)).sqrt()
    height = area / width
    center = latent[:, :2] * scale
    return torch.stack(
        [center[:, 0] - width / 2, center[:, 1] - height / 2, width, height],
        dim=1,
    )


def decode_boxes_batch(latent: torch.Tensor, areas: torch.Tensor) -> torch.Tensor:
    """Batched version of :func:`decode_boxes` with exact block areas."""

    scale = areas.sum(dim=1).sqrt().clamp_min(1.0e-6)
    rho = latent[..., 2].clamp(-RHO_CLAMP, RHO_CLAMP)
    width = (areas.clamp_min(1.0e-6) * torch.exp(rho)).sqrt()
    height = areas.clamp_min(1.0e-6) / width
    center = latent[..., :2] * scale[:, None, None]
    return torch.stack(
        [
            center[..., 0] - width / 2,
            center[..., 1] - height / 2,
            width,
            height,
        ],
        dim=-1,
    )


class SparseAttention(nn.Module):
    """Multi-head attention with a per-query top-k key mask."""

    def __init__(self, dim: int, heads: int, top_k: int | None) -> None:
        super().__init__()
        self.heads = heads
        self.top_k = top_k
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)

    def forward(self, tokens: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        batch, length, dim = tokens.shape
        head_dim = dim // self.heads
        qkv = self.qkv(tokens).reshape(batch, length, 3, self.heads, head_dim)
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        scores = query @ key.transpose(-2, -1) / math.sqrt(head_dim)
        if bias is not None:
            scores = scores + bias[:, None]
        if self.top_k is not None and self.top_k < length:
            threshold = scores.topk(int(self.top_k), dim=-1).values[..., -1:]
            scores = scores.masked_fill(scores < threshold, float("-inf"))
        attention = scores.softmax(dim=-1)
        return self.out((attention @ value).transpose(1, 2).reshape(batch, length, dim))


class BlockLayer(nn.Module):
    def __init__(self, dim: int, heads: int, top_k: int | None) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = SparseAttention(dim, heads, top_k)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, tokens: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        tokens = tokens + self.attention(self.norm1(tokens), bias)
        return tokens + self.mlp(self.norm2(tokens))


def block_features(
    areas: torch.Tensor,
    constraints: torch.Tensor,
    pins: torch.Tensor,
    b2b_edges: torch.Tensor,
) -> torch.Tensor:
    """19-ish compact per-block features matching the FloorSet conditioning."""

    n = areas.numel()
    scale = areas.sum().sqrt().clamp_min(1.0e-6)
    features = [torch.log(areas / areas.mean().clamp_min(1.0e-6))[:, None]]
    features.append(constraints[:, :5])
    pin_pull = torch.zeros(n, 2, device=areas.device, dtype=areas.dtype)
    if pins.numel() and b2b_edges.numel():
        degree = torch.zeros(n, device=areas.device, dtype=areas.dtype)
        degree.index_add_(0, b2b_edges[:, 0].long(), torch.ones(len(b2b_edges), device=areas.device))
        degree.index_add_(0, b2b_edges[:, 1].long(), torch.ones(len(b2b_edges), device=areas.device))
        features.append(degree[:, None] / degree.max().clamp_min(1.0))
    else:
        features.append(torch.zeros(n, 1, device=areas.device, dtype=areas.dtype))
    pin_mean = pins.mean(dim=0) if pins.numel() else torch.zeros(2, device=areas.device)
    pin_pull[:] = (pin_mean / scale).clamp(-2.0, 2.0)
    features.append(pin_pull)
    features.append((areas / scale.pow(2))[:, None])
    return torch.cat(features, dim=1)


class ChipBlockFlowModel(nn.Module):
    """Set transformer predicting the flow velocity in ``(cx/s, cy/s, rho)``."""

    def __init__(
        self,
        feature_dim: int = 10,
        hidden: int = 64,
        layers: int = 2,
        heads: int = 4,
        top_k: int | None = 16,
    ) -> None:
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.time = nn.Sequential(
            nn.Linear(1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.layers = nn.ModuleList(
            [BlockLayer(hidden, heads, top_k) for _ in range(layers)]
        )
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, 3)

    def forward(
        self,
        latent: torch.Tensor,
        time: torch.Tensor,
        features: torch.Tensor,
        adjacency: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tokens = self.feature(features) + self.time(time[:, None])[:, None]
        tokens = tokens + F.linear(
            latent,
            torch.eye(tokens.shape[-1], 3, device=latent.device),
        )
        bias = adjacency if adjacency is not None else None
        for layer in self.layers:
            tokens = layer(tokens, bias)
        return self.head(self.norm(tokens))


__all__ = [
    "ChipBlockFlowModel",
    "block_features",
    "decode_boxes",
    "decode_boxes_batch",
    "encode_latent",
    "encode_latents",
]
