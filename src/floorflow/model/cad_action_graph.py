"""Ordered typed attention over CAD design action stages."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from floorflow.model.official_mita_route import OfficialMiTARouteAttention

STAGES = (
    "envelope",
    "program",
    "rooms",
    "walls",
    "junctions",
    "openings",
    "furniture",
    "validation",
)

PREDECESSORS = {
    "envelope": (),
    "program": ("envelope",),
    "rooms": ("envelope", "program"),
    "walls": ("envelope", "program", "rooms"),
    "junctions": ("rooms", "walls"),
    "openings": ("rooms", "walls", "junctions"),
    "furniture": ("program", "rooms"),
    "validation": STAGES[:-1],
}

STAGE_INDEX = {name: index for index, name in enumerate(STAGES)}


@dataclass(frozen=True)
class CadActionTokens:
    tokens: dict[str, torch.Tensor]
    gates: dict[str, torch.Tensor]
    link_prior: torch.Tensor
    route_diagnostics: dict[str, dict[str, torch.Tensor]] | None = None


class CadActionGraph(nn.Module):
    """Route each CAD stage only from its declared predecessor stages."""

    def __init__(
        self,
        stage_channels: dict[str, int],
        *,
        hidden_dim: int = 96,
        heads: int = 4,
        dynamic_stage_links: bool = False,
        route_attention: bool = False,
        route_router_count: int = 8,
        route_top_k: int = 4,
        differentiable_route_budget: bool = False,
    ) -> None:
        super().__init__()
        missing = set(STAGES) - set(stage_channels)
        if missing:
            raise ValueError(f"missing stage channels: {sorted(missing)}")
        hidden_dim = int(hidden_dim)
        if hidden_dim % heads:
            heads = 1
        self.hidden_dim = hidden_dim
        self.projections = nn.ModuleDict(
            {
                name: nn.Linear(int(stage_channels[name]), hidden_dim)
                for name in STAGES
            }
        )
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            heads,
            batch_first=True,
        )
        self.gates = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, 1),
                )
                for name in STAGES
            }
        )
        self.update_scale = nn.Parameter(torch.tensor(0.01))
        prior = torch.full((len(STAGES), len(STAGES)), -4.0)
        for name, predecessors in PREDECESSORS.items():
            row = STAGE_INDEX[name]
            for predecessor in predecessors:
                prior[row, STAGE_INDEX[predecessor]] = 1.0
        self.register_buffer("link_prior", prior)
        self.link_residual = nn.Parameter(torch.zeros_like(prior))
        self.dynamic_stage_links = bool(dynamic_stage_links)
        self.dynamic_query = (
            nn.Linear(hidden_dim, hidden_dim, bias=False)
            if self.dynamic_stage_links
            else None
        )
        self.dynamic_key = (
            nn.Linear(hidden_dim, hidden_dim, bias=False)
            if self.dynamic_stage_links
            else None
        )
        self.dynamic_scale = (
            nn.Parameter(torch.tensor(0.01))
            if self.dynamic_stage_links
            else None
        )
        self.route_attention = bool(route_attention)
        self.route_router_count = max(1, int(route_router_count))
        self.route_top_k = max(1, int(route_top_k))
        self.differentiable_route_budget = bool(
            differentiable_route_budget
        )
        routed_stages = tuple(
            name for name in STAGES if PREDECESSORS[name]
        )
        self.route = (
            OfficialMiTARouteAttention(
                hidden_dim,
                num_heads=heads,
                router_count=min(self.route_router_count, len(STAGES)),
                top_k=min(self.route_top_k, len(STAGES)),
            )
            if self.route_attention
            else None
        )
        self.route_budget = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(
                        hidden_dim,
                        min(self.route_router_count, len(STAGES)),
                    ),
                )
                for name in routed_stages
            }
            if self.route_attention
            else {}
        )
        self.route_gate = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, 1),
                )
                for name in routed_stages
            }
            if self.route_attention
            else {}
        )
        self.route_scale = (
            nn.Parameter(torch.tensor(0.01))
            if self.route_attention
            else None
        )
        for budget, gate in zip(
            self.route_budget.values(),
            self.route_gate.values(),
            strict=True,
        ):
            nn.init.normal_(
                budget[-1].weight,
                mean=0.0,
                std=1.0e-3,
            )
            nn.init.zeros_(budget[-1].bias)
            nn.init.normal_(gate[-1].weight, mean=0.0, std=1.0e-3)
            nn.init.zeros_(gate[-1].bias)

    def _token(self, name: str, value: torch.Tensor) -> torch.Tensor:
        if value.ndim == 4:
            value = value.float()
            kernel = tuple(value.shape[-2:])
            value = 0.5 * (
                F.avg_pool2d(value, kernel)
                + F.max_pool2d(value, kernel)
            ).flatten(1)
        elif value.ndim == 2:
            value = value.float()
        else:
            raise ValueError(f"{name} must be [B,C] or [B,C,H,W]")
        if value.shape[1] != self.projections[name].in_features:
            raise ValueError(
                f"{name} has {value.shape[1]} channels, expected "
                f"{self.projections[name].in_features}"
            )
        return self.projections[name](value)

    def forward(self, inputs: dict[str, torch.Tensor]) -> CadActionTokens:
        raw = {
            name: self._token(name, inputs[name])
            for name in STAGES
        }
        tokens: dict[str, torch.Tensor] = {}
        gates: dict[str, torch.Tensor] = {}
        route_diagnostics = {} if self.route_attention else None
        for stage_index, name in enumerate(STAGES):
            predecessors = PREDECESSORS[name]
            current = raw[name]
            if not predecessors:
                tokens[name] = current
                gates[name] = current.new_ones(current.shape[0])
                continue
            memory = torch.stack([tokens[item] for item in predecessors], dim=1)
            row = self.link_prior[STAGE_INDEX[name]].clone()
            row = row + self.link_residual[STAGE_INDEX[name]]
            index = torch.tensor(
                [STAGE_INDEX[item] for item in predecessors],
                device=row.device,
            )
            bias = row[index][None, None, :].expand(
                current.shape[0] * self.attention.num_heads,
                1,
                -1,
            )
            if (
                self.dynamic_query is not None
                and self.dynamic_key is not None
                and self.dynamic_scale is not None
            ):
                dynamic = (
                    self.dynamic_query(current)[:, None, :]
                    * self.dynamic_key(memory)
                ).sum(dim=-1) / float(
                    current.shape[-1]
                ) ** 0.5
                dynamic = dynamic[:, None, :].expand(
                    -1,
                    self.attention.num_heads,
                    -1,
                ).reshape(
                    current.shape[0] * self.attention.num_heads,
                    1,
                    -1,
                )
                bias = (
                    bias
                    + torch.tanh(self.dynamic_scale) * dynamic
                )
            context = self.attention(
                current[:, None],
                memory,
                memory,
                attn_mask=bias,
                need_weights=False,
            )[0][:, 0]
            gate = torch.sigmoid(
                self.gates[name](torch.cat([current, context], dim=-1))
            )[:, 0]
            update = torch.tanh(self.update_scale) * gate[:, None] * context
            if (
                self.route is not None
                and self.route_scale is not None
                and route_diagnostics is not None
            ):
                available = [
                    tokens[item]
                    for item in STAGES[:stage_index]
                ]
                route_memory = torch.stack(available, dim=1)
                route_queries = torch.cat(
                    [current[:, None], route_memory],
                    dim=1,
                )
                router_count = min(
                    self.route_router_count,
                    route_queries.shape[1],
                )
                base_top_k = min(
                    self.route_top_k,
                    route_memory.shape[1],
                )
                budget_logits = torch.sigmoid(
                    self.route_budget[name](
                        torch.cat(
                            [
                                current,
                                route_memory.mean(dim=1),
                            ],
                            dim=-1,
                        )
                    )
                )[..., :router_count]
                if self.differentiable_route_budget:
                    budget_soft = (
                        0.5 + 0.5 * budget_logits
                        if base_top_k == 1
                        else 1.0 + budget_logits * (base_top_k - 1)
                    )
                else:
                    budget_soft = (
                        1.0 + budget_logits * (base_top_k - 1)
                    )
                soft_budget = (
                    budget_soft
                    if self.differentiable_route_budget
                    else None
                )
                budgets = budget_soft.detach().round().clamp(
                    1,
                    base_top_k,
                ).to(torch.long)
                routed, diagnostics = self.route(
                    route_queries,
                    route_memory,
                    router_budgets=budgets,
                    router_soft_budgets=soft_budget,
                )
                route_context = routed[:, 0]
                route_gate = torch.sigmoid(
                    self.route_gate[name](
                        torch.cat([current, route_context], dim=-1)
                    )
                )[:, 0]
                gate = gate * (0.5 + route_gate)
                budget_strength = (
                    budget_soft.mean(dim=1) / base_top_k
                )[:, None]
                update = update + (
                    torch.tanh(self.route_scale)
                    * budget_strength
                    * route_gate[:, None]
                    * route_context
                )
                route_diagnostics[name] = {
                    **diagnostics,
                    "route_budget_soft_mean": budget_soft.mean(),
                    "route_gate_mean": route_gate.mean(),
                }
            tokens[name] = current + update
            gates[name] = gate
        return CadActionTokens(
            tokens=tokens,
            gates=gates,
            link_prior=self.link_prior,
            route_diagnostics=route_diagnostics,
        )


__all__ = [
    "PREDECESSORS",
    "STAGES",
    "CadActionGraph",
    "CadActionTokens",
]
