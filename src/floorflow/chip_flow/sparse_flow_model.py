"""Graph-conditioned chip flow with learned sparse hierarchy readout."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from floorflow.chip_flow.sparse_hierarchy import (
    SparseHierarchyEncoder,
    SparseHierarchyEncoding,
)
from floorflow.chip_flow.sparse_layers import ResidualTokenBlock
from floorflow.model.dynamic_budget import allocate_group_token_budgets
from floorflow.model.official_mita_route import OfficialMiTARouteAttention


@dataclass(frozen=True)
class SparseConditionContext:
    hierarchy: SparseHierarchyEncoding
    global_state: torch.Tensor


def decoder_router_budgets(
    hierarchy: SparseHierarchyEncoding,
    router_count: int,
    base_top_k: int,
) -> torch.Tensor:
    """Allocate case-dependent budgets across active parent slots."""

    active = hierarchy.active_slot_mask[:, :router_count]
    priorities = hierarchy.slot_weights[:, :router_count]
    capacities = active.long() * int(base_top_k)
    minimum = active.long()
    complexity = hierarchy.diagnostics["complexity"].clamp(0.0, 1.0)
    extra = (
        (capacities - minimum).sum(dim=1).float()
        * (0.35 + 0.65 * complexity)
    ).round().long()
    return allocate_group_token_budgets(
        priorities,
        capacities,
        minimum.sum(dim=1) + extra,
        minimum_counts=minimum,
        maximum_counts=capacities,
    )


def sparse_regularization_terms(
    context: SparseConditionContext,
    decode_diagnostics: dict[str, torch.Tensor] | None,
    max_slots: int,
) -> dict[str, torch.Tensor]:
    """Shared budget, dead-slot, and load-balance terms."""

    diagnostics = context.hierarchy.diagnostics
    budget = diagnostics["budget_soft"].mean() / max_slots
    active_mask = context.hierarchy.active_slot_mask
    active = active_mask.float()
    active = active / active.sum(dim=1, keepdim=True).clamp_min(1.0)
    load = diagnostics["assignment_load"]
    imbalance = (load - active).square().sum(dim=1).mean()
    terms = {
        "budget": budget,
        "route_balance": imbalance,
        "dead_slot_fraction": diagnostics["dead_slot_fraction"],
    }
    if decode_diagnostics is not None and "router_load" in decode_diagnostics:
        router_load = decode_diagnostics["router_load"]
        relative = router_load * active_mask
        relative = relative / relative.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1.0e-6)
        target = active_mask.float() / active_mask.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1.0)
        terms["decode_route_balance"] = (relative - target).square().sum(
            dim=1
        ).mean()
        terms["route_balance_total"] = (
            terms["route_balance"] + terms["decode_route_balance"]
        )
    else:
        terms["route_balance_total"] = terms["route_balance"]
    return terms


class SparseHierarchyDecoder(nn.Module):
    """Decode block latents through sparse MiTA slot routing."""

    def __init__(
        self,
        feature_dim: int,
        hidden: int,
        global_dim: int,
        layers: int,
        heads: int,
        max_slots: int,
        decoder_top_k: int,
    ) -> None:
        super().__init__()
        self.max_slots = int(max_slots)
        self.decoder_top_k = int(decoder_top_k)
        self.query = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.context = nn.Linear(global_dim * 2, hidden)
        self.route = OfficialMiTARouteAttention(
            hidden,
            num_heads=heads,
            router_count=self.max_slots,
            top_k=self.decoder_top_k,
        )
        self.gate = nn.Linear(hidden, 1)
        self.layers = nn.ModuleList(
            [ResidualTokenBlock(hidden) for _ in range(layers)]
        )
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 3),
        )

    def forward(
        self,
        global_state: torch.Tensor,
        condition: torch.Tensor,
        features: torch.Tensor,
        hierarchy: SparseHierarchyEncoding,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        tokens = self.query(features.float())
        tokens = tokens + self.context(
            torch.cat([global_state, condition], dim=-1)
        )[:, None]
        active = hierarchy.active_slot_mask
        base_top_k = min(self.decoder_top_k, self.max_slots)
        router_count = min(self.max_slots, tokens.shape[1])
        active = active[:, :router_count]
        budgets = decoder_router_budgets(
            hierarchy,
            router_count,
            base_top_k,
        )
        attended, route_diagnostics = self.route(
            tokens,
            hierarchy.slot_tokens[:, :router_count],
            router_budgets=budgets,
            key_value_mask=active,
            router_mask=active,
        )
        tokens = tokens + torch.sigmoid(self.gate(attended)) * attended
        for layer in self.layers:
            tokens = layer(tokens)
        diagnostics = {
            "active_slot_count": active.sum(dim=1),
            "router_budgets": budgets,
            "top_k": route_diagnostics["router_budget_active_mean"],
            "router_budget_per_case": route_diagnostics[
                "router_budget_active_mean"
            ],
            "router_budget_mean": route_diagnostics[
                "router_budget_active_mean"
            ].mean(),
            "router_budget_min": route_diagnostics[
                "router_budget_active_min"
            ].amin(),
            "router_budget_max": route_diagnostics[
                "router_budget_active_max"
            ].amax(),
            "router_load": route_diagnostics["router_load"],
            "router_usage": route_diagnostics["router_usage"],
            "router_usage_entropy": route_diagnostics["routing_entropy"],
            "attention_entropy": route_diagnostics["attention_entropy"],
        }
        return self.head(self.norm(tokens)), diagnostics


class ChipSparseHierarchyFlowModel(nn.Module):
    """Global flow plus graph-conditioned sparse hierarchical conditioning."""

    def __init__(
        self,
        feature_dim: int = 10,
        hidden: int = 192,
        global_dim: int = 192,
        target_layers: int = 4,
        condition_layers: int = 2,
        decoder_layers: int = 4,
        heads: int = 6,
        max_slots: int = 8,
        min_slots: int = 2,
        assignment_top_k: int = 2,
        assignment_temperature: float = 1.0,
        straight_through_assignment: bool = True,
        decoder_top_k: int = 4,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden = int(hidden)
        self.global_dim = int(global_dim)
        self.max_slots = int(max_slots)
        self.condition_hierarchy = SparseHierarchyEncoder(
            self.feature_dim,
            hidden=self.global_dim,
            layers=condition_layers,
            heads=heads,
            max_slots=max_slots,
            min_slots=min_slots,
            assignment_top_k=assignment_top_k,
            assignment_temperature=assignment_temperature,
            straight_through_assignment=straight_through_assignment,
        )
        self.target_hierarchy = SparseHierarchyEncoder(
            self.feature_dim + 3,
            hidden=self.global_dim,
            layers=target_layers,
            heads=heads,
            max_slots=max_slots,
            min_slots=min_slots,
            assignment_top_k=assignment_top_k,
            assignment_temperature=assignment_temperature,
            straight_through_assignment=straight_through_assignment,
        )
        self.decoder = SparseHierarchyDecoder(
            self.feature_dim,
            self.hidden,
            self.global_dim,
            decoder_layers,
            heads,
            max_slots,
            decoder_top_k,
        )
        self.time = nn.Sequential(
            nn.Linear(1, self.hidden),
            nn.SiLU(),
            nn.Linear(self.hidden, self.hidden),
        )
        self.velocity = nn.Sequential(
            nn.Linear(2 * self.global_dim + self.hidden, self.hidden * 2),
            nn.SiLU(),
            nn.Linear(self.hidden * 2, self.hidden),
            nn.SiLU(),
            nn.Linear(self.hidden, self.global_dim),
        )

    def encode_condition_context(
        self,
        features: torch.Tensor,
        constraints: torch.Tensor | None = None,
        *,
        b2b: torch.Tensor | None = None,
        p2b: torch.Tensor | None = None,
        pins: torch.Tensor | None = None,
        n_b2b: torch.Tensor | None = None,
        n_p2b: torch.Tensor | None = None,
        n_pins: torch.Tensor | None = None,
    ) -> SparseConditionContext:
        hierarchy = self.condition_hierarchy(
            features,
            constraints,
            b2b=b2b,
            p2b=p2b,
            pins=pins,
            n_b2b=n_b2b,
            n_p2b=n_p2b,
            n_pins=n_pins,
        )
        return SparseConditionContext(hierarchy, hierarchy.global_state)

    def encode_target(
        self,
        target_latent: torch.Tensor,
        features: torch.Tensor,
        constraints: torch.Tensor | None = None,
        *,
        b2b: torch.Tensor | None = None,
        p2b: torch.Tensor | None = None,
        pins: torch.Tensor | None = None,
        n_b2b: torch.Tensor | None = None,
        n_p2b: torch.Tensor | None = None,
        n_pins: torch.Tensor | None = None,
    ) -> torch.Tensor:
        target_features = torch.cat(
            [target_latent.float(), features.float()],
            dim=-1,
        )
        hierarchy = self.target_hierarchy(
            target_features,
            constraints,
            b2b=b2b,
            p2b=p2b,
            pins=pins,
            n_b2b=n_b2b,
            n_p2b=n_p2b,
            n_pins=n_pins,
        )
        return hierarchy.global_state

    def decode_with_diagnostics(
        self,
        global_state: torch.Tensor,
        context: SparseConditionContext,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.decoder(
            global_state,
            context.global_state,
            features,
            context.hierarchy,
        )

    def sparse_regularization(
        self,
        context: SparseConditionContext,
        decode_diagnostics: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        return sparse_regularization_terms(
            context,
            decode_diagnostics,
            self.max_slots,
        )

    def forward(
        self,
        global_state: torch.Tensor,
        time: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        inputs = torch.cat(
            [global_state, condition, self.time(time[:, None])],
            dim=-1,
        )
        return self.velocity(inputs)

    @torch.no_grad()
    def sample(
        self,
        features: torch.Tensor,
        sampling_steps: int,
        constraints: torch.Tensor | None = None,
        *,
        b2b: torch.Tensor | None = None,
        p2b: torch.Tensor | None = None,
        pins: torch.Tensor | None = None,
        n_b2b: torch.Tensor | None = None,
        n_p2b: torch.Tensor | None = None,
        n_pins: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        context = self.encode_condition_context(
            features,
            constraints,
            b2b=b2b,
            p2b=p2b,
            pins=pins,
            n_b2b=n_b2b,
            n_p2b=n_p2b,
            n_pins=n_pins,
        )
        state = torch.randn(
            features.shape[0],
            self.global_dim,
            device=features.device,
            dtype=features.dtype,
            generator=generator,
        )
        for step in range(sampling_steps):
            time = torch.full(
                (features.shape[0],),
                step / sampling_steps,
                device=features.device,
                dtype=features.dtype,
            )
            state = state + self(state, time, context.global_state) / sampling_steps
        return self.decode_with_diagnostics(state, context, features)


__all__ = [
    "ChipSparseHierarchyFlowModel",
    "SparseConditionContext",
    "SparseHierarchyDecoder",
    "decoder_router_budgets",
    "sparse_regularization_terms",
]
