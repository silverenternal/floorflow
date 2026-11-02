"""Learned sparse hierarchy for chip floorplan token sets."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from floorflow.chip_flow.sparse_layers import GraphMessageBlock


@dataclass(frozen=True)
class SparseHierarchyEncoding:
    tokens: torch.Tensor
    slot_tokens: torch.Tensor
    slot_weights: torch.Tensor
    active_slot_mask: torch.Tensor
    token_assignments: torch.Tensor
    global_state: torch.Tensor
    diagnostics: dict[str, torch.Tensor]


class GraphTokenEncoder(nn.Module):
    """Encode blocks with native graph messages and explicit constraints."""

    def __init__(
        self,
        feature_dim: int,
        hidden: int,
        layers: int,
        heads: int,
    ) -> None:
        super().__init__()
        if feature_dim <= 5:
            raise ValueError("feature_dim must include five constraint columns")
        self.hidden = int(hidden)
        self.feature_input = nn.Sequential(
            nn.Linear(feature_dim - 5, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.constraint_input = nn.Linear(5, hidden)
        self.layers = nn.ModuleList(
            [GraphMessageBlock(hidden) for _ in range(layers)]
        )
        self.norm = nn.LayerNorm(hidden)
        self.heads = int(heads)

    def forward(
        self,
        features: torch.Tensor,
        constraints: torch.Tensor | None,
        b2b: torch.Tensor | None,
        p2b: torch.Tensor | None,
        pins: torch.Tensor | None,
        n_b2b: torch.Tensor | None,
        n_p2b: torch.Tensor | None,
        n_pins: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if constraints is None:
            constraints = features[..., :5]
        if constraints.shape != (features.shape[0], features.shape[1], 5):
            raise ValueError("constraints must be [B,N,5]")
        tokens = self.feature_input(features.float()[..., 5:])
        tokens = tokens + self.constraint_input(constraints.float())
        for layer in self.layers:
            tokens = layer(
                tokens,
                b2b=b2b,
                p2b=p2b,
                pins=pins,
                n_b2b=n_b2b,
                n_p2b=n_p2b,
                n_pins=n_pins,
            )
        return self.norm(tokens), graph_complexity(
            constraints,
            b2b,
            n_b2b,
        )


def graph_complexity(
    constraints: torch.Tensor,
    b2b: torch.Tensor | None,
    n_b2b: torch.Tensor | None,
) -> torch.Tensor:
    """Case-level graph density used by the learned slot-budget head."""

    batch, count, _ = constraints.shape
    constraint_density = constraints.ne(0).float().mean(dim=(1, 2))
    if b2b is None:
        edge_count = torch.zeros(batch, device=constraints.device)
    elif n_b2b is None:
        edge_count = torch.full(
            (batch,),
            float(b2b.shape[1]),
            device=constraints.device,
        )
    else:
        edge_count = n_b2b.to(constraints.device).float()
    size_fraction = constraints.new_tensor(
        (count - 21.0) / 99.0
    ).clamp(0.0, 1.0)
    edge_fraction = (edge_count / max(1, count * 3)).clamp(0.0, 1.0)
    return (
        0.6 * size_fraction
        + 0.2 * edge_fraction
        + 0.2 * constraint_density
    ).clamp(0.0, 1.0)


class LearnedSparseSlotRouter(nn.Module):
    """Route tokens to a learned, case-dependent sparse set of slots."""

    def __init__(
        self,
        hidden: int,
        max_slots: int = 8,
        min_slots: int = 2,
        assignment_top_k: int = 2,
        assignment_temperature: float = 1.0,
        straight_through_assignment: bool = True,
    ) -> None:
        super().__init__()
        if not 1 <= min_slots <= max_slots:
            raise ValueError("expected 1 <= min_slots <= max_slots")
        if not 1 <= assignment_top_k <= min_slots:
            raise ValueError(
                "assignment_top_k must not exceed the minimum active slots"
            )
        self.max_slots = int(max_slots)
        self.min_slots = int(min_slots)
        self.assignment_top_k = int(assignment_top_k)
        if assignment_temperature <= 0.0:
            raise ValueError("assignment_temperature must be positive")
        self.assignment_temperature = float(assignment_temperature)
        self.straight_through_assignment = bool(straight_through_assignment)
        self.slot_queries = nn.Parameter(
            torch.randn(self.max_slots, hidden) * 0.02
        )
        self.budget_head = nn.Linear(hidden, 1)
        nn.init.zeros_(self.budget_head.weight)
        nn.init.zeros_(self.budget_head.bias)

    def forward(
        self,
        tokens: torch.Tensor,
        complexity: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict,
    ]:
        batch, count, _ = tokens.shape
        normalized_tokens = torch.nn.functional.normalize(tokens, dim=-1)
        normalized_queries = torch.nn.functional.normalize(
            self.slot_queries,
            dim=-1,
        )
        scores = torch.einsum(
            "bnd,sd->bns",
            normalized_tokens,
            normalized_queries,
        )
        learned = torch.sigmoid(
            self.budget_head(tokens.mean(dim=1)).squeeze(-1)
        )
        continuous_budget = self.min_slots + (
            self.max_slots - self.min_slots
        ) * (0.75 * complexity + 0.25 * learned)
        budget = continuous_budget.round().long().clamp(
            self.min_slots,
            self.max_slots,
        )
        importance = torch.logsumexp(scores, dim=1)
        active = torch.zeros(
            batch,
            self.max_slots,
            device=tokens.device,
            dtype=torch.bool,
        )
        selected_slots = importance.topk(self.max_slots, dim=-1).indices
        ranks = torch.arange(self.max_slots, device=tokens.device)
        available = min(self.max_slots, count)
        active.scatter_(
            1,
            selected_slots,
            (ranks[None, :] < budget[:, None])
            & (ranks[None, :] < available),
        )
        masked_scores = scores.masked_fill(~active[:, None, :], -1.0e4)
        top_values, top_indices = masked_scores.topk(
            self.assignment_top_k,
            dim=-1,
        )
        hard_weights = torch.softmax(top_values, dim=-1)
        if self.straight_through_assignment:
            soft_weights = torch.softmax(
                masked_scores / self.assignment_temperature,
                dim=-1,
            )
            selected = torch.zeros_like(soft_weights)
            selected.scatter_(-1, top_indices, 1.0)
            soft_top = soft_weights * selected
            soft_top = soft_top / soft_top.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(1.0e-6)
            soft_top_values = torch.gather(soft_top, -1, top_indices)
            weights = (
                hard_weights
                + soft_top_values
                - soft_top_values.detach()
            )
        else:
            soft_top = hard_weights
            weights = hard_weights
        flat_indices = (
            top_indices
            + torch.arange(count, device=tokens.device)[None, :, None]
            * self.max_slots
        ).reshape(batch, count * self.assignment_top_k)
        flat_weights = weights.reshape(
            batch,
            count * self.assignment_top_k,
        ).to(tokens.dtype)
        assignments = torch.zeros(
            batch,
            count * self.max_slots,
            device=tokens.device,
            dtype=tokens.dtype,
        )
        assignments.scatter_(1, flat_indices, flat_weights)
        assignments = assignments.reshape(batch, count, self.max_slots)
        mass = assignments.sum(dim=1)
        normalized = assignments / mass[:, None, :].clamp_min(1.0e-6)
        slot_tokens = torch.bmm(
            normalized.transpose(1, 2),
            tokens,
        ) * active[:, :, None]
        slot_weights = mass / mass.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
        slot_weights = slot_weights * active
        slot_weights = slot_weights / slot_weights.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1.0e-6)
        diagnostics = {
            "active_slot_count": active.sum(dim=1),
            "slot_count_mean": active.sum(dim=1).float().mean(),
            "slot_count_min": active.sum(dim=1).amin(),
            "slot_count_max": active.sum(dim=1).amax(),
            "assignment_top_k": torch.tensor(
                self.assignment_top_k,
                device=tokens.device,
            ),
            "assignment_edges": torch.tensor(
                batch * count * self.assignment_top_k,
                device=tokens.device,
            ),
            "slot_usage": active.float().mean(dim=0),
            "budget_soft": continuous_budget,
            "assignment_load": mass.mean(dim=0)
            / mass.mean().clamp_min(1.0e-6),
            "assignment_entropy": -(
                soft_top.clamp_min(1.0e-8)
                * soft_top.clamp_min(1.0e-8).log()
            ).sum(dim=-1).mean(),
            "dead_slot_fraction": (~active.any(dim=0)).float().mean(),
            "router_load": mass / mass.sum(dim=1, keepdim=True).clamp_min(
                1.0e-6
            ),
        }
        return slot_tokens, slot_weights, active, assignments, diagnostics


class SparseHierarchyEncoder(nn.Module):
    """Graph encoder plus learned sparse slot hierarchy."""

    def __init__(
        self,
        feature_dim: int,
        hidden: int = 192,
        layers: int = 2,
        heads: int = 6,
        max_slots: int = 8,
        min_slots: int = 2,
        assignment_top_k: int = 2,
        assignment_temperature: float = 1.0,
        straight_through_assignment: bool = True,
    ) -> None:
        super().__init__()
        self.graph = GraphTokenEncoder(feature_dim, hidden, layers, heads)
        self.router = LearnedSparseSlotRouter(
            hidden,
            max_slots=max_slots,
            min_slots=min_slots,
            assignment_top_k=assignment_top_k,
            assignment_temperature=assignment_temperature,
            straight_through_assignment=straight_through_assignment,
        )
        self.pool = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

    def forward(
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
    ) -> SparseHierarchyEncoding:
        if constraints is None:
            constraints = features[..., :5]
        tokens, complexity = self.graph(
            features,
            constraints,
            b2b,
            p2b,
            pins,
            n_b2b,
            n_p2b,
            n_pins,
        )
        slots, weights, active, assignments, diagnostics = self.router(
            tokens,
            complexity,
        )
        pooled = torch.cat([tokens.mean(dim=1), tokens.amax(dim=1)], dim=-1)
        global_state = self.pool(pooled)
        diagnostics["complexity"] = complexity
        diagnostics["constraint_density"] = constraints.ne(0).float().mean(
            dim=(1, 2)
        )
        return SparseHierarchyEncoding(
            tokens=tokens,
            slot_tokens=slots,
            slot_weights=weights,
            active_slot_mask=active,
            token_assignments=assignments,
            global_state=global_state,
            diagnostics=diagnostics,
        )


__all__ = [
    "GraphTokenEncoder",
    "LearnedSparseSlotRouter",
    "SparseHierarchyEncoder",
    "SparseHierarchyEncoding",
    "graph_complexity",
]
