"""LayoutBridge-derived structure refinement blocks.

Adapted for floorplan flow matching from:
https://github.com/IndoorMapping-Lab/LayoutBridge
commit e02974a, MIT license.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DualStreamStructureInjection(nn.Module):
    """DDI-style structure/contour injection adapted to feature maps."""

    def __init__(self, hidden_channels: int, guidance_channels: int) -> None:
        super().__init__()
        inner = max(32, hidden_channels // 2)
        self.guidance_projection = (
            nn.Conv2d(guidance_channels, hidden_channels, 1)
            if guidance_channels != hidden_channels
            else nn.Identity()
        )
        self.structure_gate = nn.Sequential(
            nn.Conv2d(hidden_channels, inner, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(inner, hidden_channels, 3, padding=1),
            nn.Sigmoid(),
        )
        self.structure_projection = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1)
        self.regularity_encoder = nn.Sequential(
            nn.Conv2d(hidden_channels, inner, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(inner, inner, 3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.time_film = nn.Sequential(
            nn.Linear(inner + 1, inner),
            nn.SiLU(),
            nn.Linear(inner, hidden_channels * 2),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_channels * 2 + 1, inner, 1),
            nn.SiLU(),
            nn.Conv2d(inner, hidden_channels, 1),
            nn.Sigmoid(),
        )
        self.residual_scale = nn.Parameter(torch.zeros(1, hidden_channels, 1, 1))

    def forward(
        self,
        features: torch.Tensor,
        guidance: torch.Tensor,
        time_value: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        guidance = self.guidance_projection(guidance)
        gate = self.structure_gate(features)
        structure = features + gate * self.structure_projection(features)
        regularity = self.regularity_encoder(guidance).flatten(1)
        time_column = time_value.float()[:, None]
        gamma, beta = self.time_film(torch.cat([regularity, time_column], dim=1)).chunk(2, dim=1)
        normalized = F.layer_norm(structure, structure.shape[1:])
        regularized = normalized * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
        time_map = time_value.float()[:, None, None, None].expand(
            -1,
            1,
            structure.shape[-2],
            structure.shape[-1],
        )
        weight = self.fusion(torch.cat([features, structure, time_map], dim=1))
        mixed = weight * regularized + (1.0 - weight) * structure
        return features + self.residual_scale * (mixed - features), {
            "ddi_gate": gate.mean(),
            "ddi_fusion": weight.mean(),
        }


def _spatial_gradient(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    grad_x = F.pad(value[..., :, 1:] - value[..., :, :-1], (0, 1, 0, 0))
    grad_y = F.pad(value[..., 1:, :] - value[..., :-1, :], (0, 0, 0, 1))
    return grad_x, grad_y


def _spatial_divergence(flow_x: torch.Tensor, flow_y: torch.Tensor) -> torch.Tensor:
    div_x = F.pad(flow_x[..., :, 1:] - flow_x[..., :, :-1], (1, 0, 0, 0))
    div_y = F.pad(flow_y[..., 1:, :] - flow_y[..., :-1, :], (0, 0, 1, 0))
    return div_x + div_y


def anisotropic_operator(value: torch.Tensor, tensor: tuple[torch.Tensor, ...]) -> torch.Tensor:
    a_xx, a_xy, a_yy = tensor
    grad_x, grad_y = _spatial_gradient(value)
    flow_x = a_xx * grad_x + a_xy * grad_y
    flow_y = a_xy * grad_x + a_yy * grad_y
    return _spatial_divergence(flow_x, flow_y)


class AnisotropicVelocityRefinement(nn.Module):
    """Positive-definite 2D tensor field for direction-aware velocity updates."""

    def __init__(self, value_channels: int, feature_channels: int, hidden_channels: int) -> None:
        super().__init__()
        hidden = max(32, int(hidden_channels))
        self.value_channels = int(value_channels)
        self.net = nn.Sequential(
            nn.Conv2d(feature_channels + 1, hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, 3, 3, padding=1),
        )
        self.strength = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        value: torch.Tensor,
        features: torch.Tensor,
        time_value: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if value.shape[1] != self.value_channels:
            raise ValueError(f"expected {self.value_channels} value channels, got {value.shape[1]}")
        time_map = time_value.float()[:, None, None, None].expand(
            -1,
            1,
            features.shape[-2],
            features.shape[-1],
        )
        theta, a_perp, a_parallel = self.net(torch.cat([features, time_map], dim=1)).chunk(3, dim=1)
        lambda_perp = F.softplus(a_perp) + 1.0e-4
        lambda_parallel = lambda_perp + F.softplus(a_parallel)
        cosine, sine = torch.cos(theta), torch.sin(theta)
        a_xx = lambda_perp + (lambda_parallel - lambda_perp) * cosine.square()
        a_yy = lambda_perp + (lambda_parallel - lambda_perp) * sine.square()
        a_xy = (lambda_parallel - lambda_perp) * sine * cosine
        correction = anisotropic_operator(value, (a_xx, a_xy, a_yy))
        weighted = self.strength.tanh() * time_value[:, None, None, None] * correction
        return value + weighted, {"anisotropic_strength": self.strength.detach()}


def laplacian_filter(value: torch.Tensor) -> torch.Tensor:
    channels = value.shape[1]
    kernel = value.new_tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
    kernel = kernel.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    return F.conv2d(value, kernel, padding=1, groups=channels)


def reconstruction_frequency_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    error = prediction - target
    low = F.avg_pool2d(error, 3, stride=1, padding=1).pow(2).mean()
    high = laplacian_filter(error).pow(2).mean()
    return low, high
