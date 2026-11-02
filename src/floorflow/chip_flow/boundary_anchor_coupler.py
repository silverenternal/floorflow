"""Boundary-code-only output correction for chip flow trajectories."""

from __future__ import annotations

import torch
from torch import nn

from floorflow.chip_flow.model import decode_boxes_batch


def _boundary_error(
    latent: torch.Tensor,
    areas: torch.Tensor,
    boundary_mask: torch.Tensor,
) -> torch.Tensor:
    boxes = decode_boxes_batch(latent.float(), areas.float())
    x0, y0, width, height = boxes.unbind(dim=-1)
    x1 = x0 + width
    y1 = y0 + height
    error = (
        boundary_mask[..., 0] * (x0 - x0.amin(dim=1)[:, None]).abs()
        + boundary_mask[..., 1] * (x1 - x1.amax(dim=1)[:, None]).abs()
        + boundary_mask[..., 2] * (y1 - y1.amax(dim=1)[:, None]).abs()
        + boundary_mask[..., 3] * (y0 - y0.amin(dim=1)[:, None]).abs()
    )
    return error


class BoundaryAnchorCoupler(nn.Module):
    """Correct decoded endpoint geometry only for boundary-coded blocks."""

    def __init__(self, hidden: int, *, bottleneck: int | None = None) -> None:
        super().__init__()
        if hidden <= 0:
            raise ValueError("hidden must be positive")
        width = int(bottleneck or max(16, hidden // 4))
        self.hidden = int(hidden)
        self.bottleneck = width
        self.encoder = nn.Sequential(
            nn.Linear(hidden + 8, width),
            nn.SiLU(),
            nn.Linear(width, 3),
        )
        nn.init.normal_(self.encoder[-1].weight, std=0.02)
        nn.init.zeros_(self.encoder[-1].bias)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        block_tokens: torch.Tensor,
        block_state: torch.Tensor,
        boundary_mask: torch.Tensor,
        time: torch.Tensor,
        areas: torch.Tensor | None = None,
        constraints: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, count, hidden = block_tokens.shape
        if hidden != self.hidden:
            raise ValueError("block token hidden size mismatch")
        if block_state.shape != (batch, count, 3):
            raise ValueError("block state must have shape (batch, n, 3)")
        if boundary_mask.shape != (batch, count, 4):
            raise ValueError("boundary mask must have shape (batch, n, 4)")
        if time.shape != (batch,):
            raise ValueError("time must have shape (batch,)")
        boundary = boundary_mask.to(block_tokens.dtype)
        active = boundary_mask.any(dim=-1)
        context = torch.cat(
            [
                block_tokens,
                block_state.float().to(block_tokens.dtype),
                boundary,
                time.float()[:, None, None].expand(-1, count, 1),
            ],
            dim=-1,
        )
        residual = self.gate * self.encoder(context)
        residual = residual * active[..., None]
        accepted = active
        if areas is not None:
            if areas.shape != block_state.shape[:2]:
                raise ValueError("areas must match boundary block axes")
            current_error = _boundary_error(
                block_state,
                areas,
                boundary,
            )
            proposed_error = _boundary_error(
                block_state + residual,
                areas,
                boundary,
            )
            accepted = active & proposed_error.lt(current_error)
            residual = residual * accepted[..., None].to(residual.dtype)
        if constraints is not None:
            if constraints.shape != (batch, count, 5):
                raise ValueError("constraints must have shape [B,N,5]")
            preplaced = constraints[..., 1].ne(0)
            fixed = constraints[..., 0].ne(0)
            residual = torch.cat(
                [
                    residual[..., :2]
                    * (~preplaced)[..., None].to(residual.dtype),
                    residual[..., 2:]
                    * (~(fixed | preplaced))[..., None].to(residual.dtype),
                ],
                dim=-1,
            )
        return residual, {
            "boundary_anchor_active_fraction": active.float().mean(),
            "boundary_anchor_accept_fraction": accepted.float().mean(),
            "boundary_anchor_residual_norm": residual.detach().norm(dim=-1),
        }


__all__ = ["BoundaryAnchorCoupler"]
