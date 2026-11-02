"""Sparse coarse-control attention for masked guided CFM."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from floorflow.model.dynamic_budget import allocate_group_token_budgets
from floorflow.model.official_mita_route import OfficialMiTARouteAttention


class SparseControlAttention(nn.Module):
    """Cross-attend current design tokens to sparse coarse-layout tokens."""

    def __init__(
        self,
        channels: int,
        *,
        attention_type: str = "mita",
        heads: int = 4,
        router_count: int = 16,
        top_k: int = 8,
        token_grid: int = 16,
    ) -> None:
        super().__init__()
        if attention_type not in {"mita", "mha", "adaptive"}:
            raise ValueError("attention_type must be mita, mha, or adaptive")
        if channels % heads:
            heads = 1
        self.channels = int(channels)
        self.token_grid = int(token_grid)
        self.router_count = int(router_count)
        self.base_top_k = int(top_k)
        self.control_projection = nn.Sequential(
            nn.Linear(channels + 2, channels),
            nn.LayerNorm(channels),
            nn.SiLU(),
        )
        self.state_projection = nn.Sequential(
            nn.Linear(channels, channels),
            nn.LayerNorm(channels),
        )
        self.attention_type = attention_type
        self.attention = (
            OfficialMiTARouteAttention(
                channels,
                num_heads=heads,
                router_count=router_count,
                top_k=top_k,
            )
            if attention_type in {"mita", "adaptive"}
            else nn.MultiheadAttention(
                channels,
                heads,
                batch_first=True,
            )
        )
        self.alternate_attention = (
            nn.MultiheadAttention(channels, heads, batch_first=True)
            if attention_type == "adaptive"
            else None
        )
        self.adaptive_scale = nn.Parameter(torch.ones(1))
        self.adaptive_bias = nn.Parameter(torch.zeros(1))
        self.output = nn.Sequential(
            nn.Linear(channels, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
        )
        self.route_prior = nn.Sequential(
            nn.Linear(3, channels),
            nn.SiLU(),
            nn.Linear(channels, self.router_count),
        )
        nn.init.zeros_(self.route_prior[-1].weight)
        nn.init.zeros_(self.route_prior[-1].bias)
        self.route_prior_scale = nn.Parameter(torch.full((1,), 0.1))
        self.scale = nn.Parameter(torch.full((1,), 0.05))
        self.last_diagnostics: dict[str, torch.Tensor] = {}

    def forward(
        self,
        state: torch.Tensor,
        control: torch.Tensor,
        mask: torch.Tensor,
        footprint: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if footprint.ndim == 3:
            footprint = footprint[:, None]
        if mask.ndim == 3:
            mask = mask[:, None]
        if mask.shape[1] != 1:
            mask = mask.float().amax(dim=1, keepdim=True)
        size = (self.token_grid, self.token_grid)
        state_tokens = F.adaptive_avg_pool2d(state, size).flatten(2).transpose(1, 2)
        control_tokens = F.adaptive_avg_pool2d(control, size).flatten(2).transpose(1, 2)
        mask_tokens = F.adaptive_avg_pool2d(mask.float(), size).flatten(2).transpose(1, 2)
        footprint_tokens = F.adaptive_avg_pool2d(
            footprint.float(),
            size,
        ).flatten(2).transpose(1, 2)
        query = self.state_projection(state_tokens)
        control_energy = control_tokens.abs().mean(dim=-1, keepdim=True)
        route_prior = torch.tanh(
            self.route_prior(
                torch.cat(
                    [mask_tokens, control_energy, footprint_tokens],
                    dim=-1,
                )
            )
        ) * self.route_prior_scale.to(mask_tokens.dtype)
        memory = self.control_projection(
            torch.cat([control_tokens, mask_tokens, footprint_tokens], dim=-1)
        )
        base = min(self.base_top_k, memory.shape[1])
        if self.attention_type in {"mita", "adaptive"}:
            router_count = min(self.router_count, query.shape[1])
            route_prior_used = route_prior[:, :, :router_count]
            prior_probability = route_prior_used.softmax(dim=-1).mean(dim=1)
            capacities = torch.full(
                (query.shape[0], router_count),
                memory.shape[1],
                device=query.device,
                dtype=torch.long,
            )
            router_budgets = allocate_group_token_budgets(
                prior_probability,
                capacities,
                torch.full(
                    (query.shape[0],),
                    router_count * base,
                    device=query.device,
                    dtype=torch.long,
                ),
                minimum_counts=torch.full_like(
                    capacities,
                    max(1, int(base * 0.5)),
                ),
                maximum_counts=torch.minimum(
                    capacities,
                    torch.full_like(
                        capacities,
                        max(base, int(base * 1.5)),
                    ),
                ),
            )
            attended, diagnostics = self.attention(
                query,
                memory,
                router_prior=route_prior_used,
                router_budgets=router_budgets,
            )
            routing = diagnostics["routing"]
            expert_probability = routing.mean(dim=(0, 1, 2))
            route_balance = (
                router_count * expert_probability.square().sum() - 1.0
            ).clamp_min(0.0)
            diagnostics["route_balance_loss"] = route_balance
            diagnostics["router_budget_mean"] = diagnostics["top_k"]
            if self.attention_type == "adaptive":
                alternate = self.alternate_attention(
                    query,
                    memory,
                    memory,
                    need_weights=False,
                )[0]
                coverage = mask.float().mean(dim=(1, 2, 3))
                mha_weight = torch.sigmoid(
                    self.adaptive_bias.to(query.dtype)
                    - self.adaptive_scale.to(query.dtype) * coverage
                )
                attended = (
                    (1.0 - mha_weight[:, None, None]) * attended
                    + mha_weight[:, None, None] * alternate
                )
                diagnostics["adaptive_mha_weight"] = mha_weight.mean()
                diagnostics["adaptive_mha_weight_per_sample"] = mha_weight
                diagnostics["adaptive_control_coverage"] = coverage.mean()
            self.last_diagnostics = diagnostics
        else:
            attended = self.attention(
                query,
                memory,
                memory,
                need_weights=False,
            )[0]
            diagnostics = {"router_budget_mean": torch.tensor(base, device=query.device)}
        attended = self.output(attended).transpose(1, 2).reshape(
            state.shape[0],
            self.channels,
            *size,
        )
        attended = F.interpolate(
            attended,
            size=state.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        covered = F.adaptive_avg_pool2d(mask.float(), (1, 1)).mean()
        diagnostics = dict(diagnostics)
        diagnostics["mask_coverage"] = covered
        diagnostics["router_prior_abs"] = route_prior.abs().mean()
        return self.scale.to(state.dtype) * attended * footprint, diagnostics


class MaskedControlFlow(nn.Module):
    """Wrap a CFM backbone with explicit sparse-control attention."""

    def __init__(
        self,
        base,
        *,
        attention_type: str = "mita",
    ) -> None:
        super().__init__()
        self.base = base
        self.target_channels = base.target_channels
        self.sparse_attention = SparseControlAttention(
            base.target_channels,
            attention_type=attention_type,
        )
        self.last_sparse_diagnostics: dict[str, torch.Tensor] = {}

    def forward_with_aux(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        footprint: torch.Tensor,
        *,
        floor_ids: torch.Tensor | None = None,
        control: torch.Tensor,
        control_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        velocity, auxiliary = self.base.forward_with_aux(
            x_t,
            t,
            footprint,
            floor_ids=floor_ids,
        )
        residual, diagnostics = self.sparse_attention(
            x_t,
            control,
            control_mask,
            footprint,
        )
        self.last_sparse_diagnostics = diagnostics
        auxiliary = dict(auxiliary)
        auxiliary["sparse_control"] = diagnostics
        return velocity + residual, auxiliary


@torch.no_grad()
def sample_sparse_guided_floorplans(
    model: MaskedControlFlow,
    footprint: torch.Tensor,
    control: torch.Tensor,
    control_mask: torch.Tensor,
    *,
    steps: int = 4,
    floor_ids: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
) -> torch.Tensor:
    if footprint.ndim == 3:
        footprint = footprint[:, None]
    state = (
        control
        if initial_state is None
        else initial_state
    ).to(footprint.dtype) * footprint.float()
    steps = max(1, int(steps))
    dt = 1.0 / steps
    for step in range(steps):
        time_value = torch.full(
            (footprint.shape[0],),
            step * dt,
            device=footprint.device,
            dtype=footprint.dtype,
        )
        velocity, _auxiliary = model.forward_with_aux(
            state,
            time_value,
            footprint,
            floor_ids=floor_ids,
            control=control,
            control_mask=control_mask,
        )
        state = (state + dt * velocity) * footprint.float()
    return state


__all__ = [
    "MaskedControlFlow",
    "SparseControlAttention",
    "sample_sparse_guided_floorplans",
]
