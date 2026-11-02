"""Port-to-block feedback for chip placement trajectories."""

from __future__ import annotations

import torch
from torch import nn


class PortConsensusCoupler(nn.Module):
    """Pool predicted L3 port offsets and inject them into L2 blocks."""

    def __init__(
        self,
        hidden: int,
        *,
        bottleneck: int | None = None,
        rho_trust_radius: float = 0.25,
        position_trust_radius: float = 0.25,
    ) -> None:
        super().__init__()
        if hidden <= 0:
            raise ValueError("hidden must be positive")
        width = int(bottleneck or max(16, hidden // 4))
        self.hidden = int(hidden)
        self.bottleneck = width
        if rho_trust_radius < 0.0:
            raise ValueError("port-consensus rho trust radius must be non-negative")
        if position_trust_radius < 0.0:
            raise ValueError(
                "port-consensus position trust radius must be non-negative"
            )
        self.rho_trust_radius = float(rho_trust_radius)
        self.position_trust_radius = float(position_trust_radius)
        self.port_encoder = nn.Sequential(
            nn.Linear(2, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.state_encoder = nn.Linear(3, width)
        self.time_encoder = nn.Linear(1, width)
        self.reduce = nn.Sequential(
            nn.Linear(hidden + 3 * width, width),
            nn.SiLU(),
            nn.Linear(width, 3),
        )
        nn.init.normal_(self.reduce[-1].weight, std=0.02)
        nn.init.zeros_(self.reduce[-1].bias)
        self.gate = nn.Parameter(torch.zeros(1))
        self.geometry_scale = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        block_tokens: torch.Tensor,
        block_state: torch.Tensor,
        port_prediction: torch.Tensor,
        incidence: torch.Tensor,
        port_mask: torch.Tensor,
        time: torch.Tensor,
        pin_positions: torch.Tensor | None = None,
        pin_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, count, hidden = block_tokens.shape
        port_count = port_prediction.shape[1]
        if hidden != self.hidden:
            raise ValueError("block token hidden size mismatch")
        if block_state.shape != (batch, count, 3):
            raise ValueError("block state must have shape (batch, n, 3)")
        if port_prediction.shape != (batch, port_count, 2):
            raise ValueError("port prediction must have shape (batch, p, 2)")
        if incidence.shape != (batch, count, port_count):
            raise ValueError("port incidence must have shape (batch, n, p)")
        if port_mask.shape != (batch, port_count):
            raise ValueError("port mask must have shape (batch, p)")
        if time.shape != (batch,):
            raise ValueError("time must have shape (batch,)")

        weights = (
            incidence.to(block_tokens.dtype)
            * port_mask[:, None, :].to(block_tokens.dtype)
        )
        denominator = weights.sum(dim=-1, keepdim=True)
        pooled = torch.bmm(weights, port_prediction.float())
        pooled = pooled / denominator.clamp_min(1.0)
        has_port = denominator.squeeze(-1).gt(0.0)
        context = torch.cat(
            [
                self.port_encoder(pooled.to(block_tokens.dtype)),
                self.state_encoder(block_state.float()).to(block_tokens.dtype),
                self.time_encoder(time.float()[:, None])[:, None].expand(
                    -1,
                    count,
                    -1,
                ),
            ],
            dim=-1,
        )
        residual = self.reduce(
            torch.cat([block_tokens, context], dim=-1)
        )
        residual = self.gate * residual * has_port[..., None]
        if pin_positions is not None:
            if pin_positions.shape != port_prediction.shape:
                raise ValueError("pin positions must match port prediction")
            if pin_mask is None:
                pin_mask = torch.ones_like(port_mask)
            if pin_mask.shape != port_mask.shape:
                raise ValueError("pin mask must match port mask")
            geometry_weights = (
                weights
                * pin_mask[:, None, :].to(weights.dtype)
            )
            implied_center = torch.bmm(
                geometry_weights,
                pin_positions.float() + port_prediction.float(),
            ) / geometry_weights.sum(dim=-1)[..., None].clamp_min(1.0)
            geometry = torch.cat(
                [
                    implied_center - block_state[..., :2].float(),
                    geometry_weights.new_zeros((*has_port.shape, 1)),
                ],
                dim=-1,
            )
            residual = residual + (
                torch.tanh(self.geometry_scale)
                * geometry.to(residual.dtype)
                * has_port[..., None]
            )
        residual = torch.cat(
            [
                residual[..., :2].clamp(
                    -self.position_trust_radius,
                    self.position_trust_radius,
                ),
                residual[..., 2:].clamp(
                    -self.rho_trust_radius,
                    self.rho_trust_radius,
                ),
            ],
            dim=-1,
        )
        return residual, {
            "port_consensus_active_fraction": has_port.float().mean(),
            "port_consensus_residual_norm": residual.detach().norm(dim=-1),
            "port_consensus_geometry_scale": self.geometry_scale.detach(),
        }


__all__ = ["PortConsensusCoupler"]
