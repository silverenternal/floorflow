"""Group-moment conditioning for per-block chip trajectories."""

from __future__ import annotations

import math

import torch
from torch import nn

GROUP_MOMENT_DIM = 7


class GroupMomentCoupler(nn.Module):
    """Inject predicted group extent/span moments into block residuals."""

    def __init__(self, hidden: int, *, width: int | None = None) -> None:
        super().__init__()
        if hidden <= 0:
            raise ValueError("hidden must be positive")
        self.hidden = int(hidden)
        self.width = int(width or max(16, hidden // 8))
        self.moment_norm = nn.LayerNorm(GROUP_MOMENT_DIM)
        self.moment_encoder = nn.Linear(GROUP_MOMENT_DIM, self.width)
        self.token_encoder = nn.Linear(hidden, self.width)
        self.state_encoder = nn.Linear(3, self.width)
        self.time_encoder = nn.Linear(1, self.width)
        self.residual = nn.Sequential(
            nn.LayerNorm(6 * self.width),
            nn.Linear(6 * self.width, self.width),
            nn.SiLU(),
            nn.Linear(self.width, 3),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        # The residual head is zero-initialized, so a unit gate preserves exact
        # identity while avoiding a multiplicative zero-gradient deadlock
        # between the gate and the final residual layer.
        self.gate = nn.Parameter(torch.ones(1))
        self.commit_bias = nn.Parameter(torch.tensor(0.5))
        self.commit_log_slope = nn.Parameter(torch.tensor(math.log(8.0)))

    def commit_weight(self, time: torch.Tensor) -> torch.Tensor:
        """Return a smooth early-explore to late-commit schedule."""

        slope = self.commit_log_slope.float().exp()
        return torch.sigmoid(slope * (time.float() - self.commit_bias.float()))

    def forward(
        self,
        block_tokens: torch.Tensor,
        block_state: torch.Tensor,
        group_moments: torch.Tensor,
        membership: torch.Tensor,
        structural_mask: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        batch, count, hidden = block_tokens.shape
        slots = group_moments.shape[1]
        if hidden != self.hidden:
            raise ValueError("block token hidden size mismatch")
        if block_state.shape != (batch, count, 3):
            raise ValueError("block state must have shape (batch, blocks, 3)")
        if group_moments.shape != (batch, slots, GROUP_MOMENT_DIM):
            raise ValueError("group moments have the wrong shape")
        if membership.shape != (batch, count, slots):
            raise ValueError("membership must match group moments")
        if structural_mask.shape != (batch, slots):
            raise ValueError("structural mask must match group moments")
        if time.shape != (batch,):
            raise ValueError("time must have shape (batch,)")

        code = self.moment_encoder(self.moment_norm(group_moments.float()))
        weights = membership.to(code.dtype) * structural_mask[:, None].to(code.dtype)
        if slots != GROUP_MOMENT_DIM + 2:
            raise ValueError("group moments require four cluster, one MIB and four boundary slots")
        typed_weights = (
            weights[..., :4],
            weights[..., 4:5],
            weights[..., 5:9],
        )
        typed_codes = (
            code[:, :4],
            code[:, 4:5],
            code[:, 5:9],
        )
        pooled = []
        for typed, typed_code in zip(typed_weights, typed_codes, strict=True):
            mass = typed.sum(dim=2, keepdim=True)
            value = torch.bmm(typed, typed_code)
            pooled.append(value / mass.clamp_min(1.0))
        token_code = self.token_encoder(block_tokens.float())
        state_code = self.state_encoder(block_state.float())
        time_code = self.time_encoder(time.float()[:, None])
        time_code = time_code[:, None].expand(-1, count, -1)
        correction = self.residual(
            torch.cat(
                [token_code, state_code, *pooled, time_code],
                dim=-1,
            )
        )
        commit = self.commit_weight(time)[:, None, None]
        return (self.gate * commit * correction).to(block_tokens.dtype)


class NativeGroupMomentModule(GroupMomentCoupler):
    """Native moment-state head and block-conditioning operator."""

    def __init__(self, hidden: int, *, width: int | None = None) -> None:
        super().__init__(hidden, width=width)
        self.moment_state_norm = nn.LayerNorm(GROUP_MOMENT_DIM)
        self.moment_state_encoder = nn.Linear(GROUP_MOMENT_DIM, self.width)
        self.structural_token_encoder = nn.Linear(hidden, self.width)
        self.native_residual = nn.Sequential(
            nn.LayerNorm(4 * self.width),
            nn.Linear(4 * self.width, self.width),
            nn.SiLU(),
            nn.Linear(self.width, GROUP_MOMENT_DIM),
        )
        nn.init.zeros_(self.native_residual[-1].weight)
        nn.init.zeros_(self.native_residual[-1].bias)

    def predict(
        self,
        structural_tokens: torch.Tensor,
        structural_state: torch.Tensor,
        moment_state: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        batch, slots, _hidden = structural_tokens.shape
        if structural_state.shape != (batch, slots, 3):
            raise ValueError("structural state must have shape (batch, slots, 3)")
        if moment_state.shape != (batch, slots, GROUP_MOMENT_DIM):
            raise ValueError("moment state must have shape (batch, slots, 7)")
        time_code = self.time_encoder(time.float()[:, None])
        time_code = time_code[:, None].expand(-1, slots, -1)
        delta = self.native_residual(
            torch.cat(
                [
                    self.structural_token_encoder(structural_tokens.float()),
                    self.state_encoder(structural_state.float()),
                    self.moment_state_encoder(
                        self.moment_state_norm(moment_state.float())
                    ),
                    time_code,
                ],
                dim=-1,
            )
        )
        return moment_state + delta.to(moment_state.dtype)


def apply_native_group_moment(
    module: NativeGroupMomentModule,
    block_tokens: torch.Tensor,
    block_state: torch.Tensor,
    structural_tokens: torch.Tensor,
    structural_state: torch.Tensor,
    moment_state: torch.Tensor,
    membership: torch.Tensor,
    structural_mask: torch.Tensor,
    time: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Predict native moments and return their gated block correction."""

    moments = module.predict(
        structural_tokens,
        structural_state,
        moment_state,
        time,
    )
    moments = moments * structural_mask[..., None].to(moments.dtype)
    residual = module(
        block_tokens,
        block_state,
        moments,
        membership,
        structural_mask,
        time,
    )
    return moments, residual


__all__ = [
    "GROUP_MOMENT_DIM",
    "GroupMomentCoupler",
    "NativeGroupMomentModule",
    "apply_native_group_moment",
]
