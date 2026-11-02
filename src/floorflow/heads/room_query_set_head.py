"""Explicit query existence, room-count, and per-class cardinality heads."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from floorflow.heads.room_query_set_objective import (
    _one_to_one_query_targets,
    room_query_set_loss,
)
from floorflow.heads.room_slot_state import poisson_binomial_log_probability
from floorflow.model.official_mita_route import OfficialMiTARouteAttention


class RoomQuerySetHead(nn.Module):
    """Predict query presence and global set cardinality from image evidence."""

    def __init__(
        self,
        query_dim: int,
        feature_channels: int,
        num_room_classes: int,
        num_queries: int,
        *,
        hidden_dim: int = 64,
        pool_size: int = 48,
        attention_layers: int = 0,
        attention_heads: int = 4,
        routing_attention: bool = False,
        route_router_count: int = 12,
        route_top_k: int = 4,
        class_count_prior: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.num_room_classes = int(num_room_classes)
        self.num_queries = int(num_queries)
        self.pool_size = max(1, int(pool_size))
        query_dim = int(query_dim)
        hidden_dim = int(hidden_dim)
        if hidden_dim % int(attention_heads):
            raise ValueError("query-set hidden dim must be divisible by heads")
        self.feature_projection = nn.Conv2d(
            int(feature_channels),
            hidden_dim,
            1,
        )
        self.presence = nn.Sequential(
            nn.Linear(
                (hidden_dim if int(attention_layers) > 0 else query_dim) + 3,
                hidden_dim,
            ),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.count = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.num_queries + 1),
        )
        self.class_count = nn.Linear(
            hidden_dim * 2,
            self.num_room_classes * (self.num_queries + 1),
        )
        self.presence_count_scale = nn.Parameter(torch.tensor(0.05))
        self.query_projection = (
            nn.Linear(query_dim, hidden_dim)
            if int(attention_layers) > 0
            else None
        )
        self.set_attention = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    hidden_dim,
                    int(attention_heads),
                    batch_first=True,
                )
                for _ in range(int(attention_layers))
            ]
        )
        self.routing_attention = bool(
            routing_attention and int(attention_layers) > 0
        )
        self.route_router_count = max(1, int(route_router_count))
        self.route_top_k = max(1, int(route_top_k))
        self.route_attention = (
            nn.ModuleList(
                [
                    OfficialMiTARouteAttention(
                        hidden_dim,
                        num_heads=int(attention_heads),
                        router_count=min(
                            self.route_router_count,
                            int(num_queries),
                        ),
                        top_k=min(self.route_top_k, int(num_queries)),
                    )
                    for _ in range(int(attention_layers))
                ]
            )
            if self.routing_attention
            else nn.ModuleList()
        )
        self.route_budget = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim // 2),
                    nn.SiLU(),
                    nn.Linear(
                        hidden_dim // 2,
                        min(self.route_router_count, int(num_queries)),
                    ),
                )
                for _ in range(len(self.route_attention))
            ]
        )
        self.route_gate = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, 1),
                )
                for _ in range(len(self.route_attention))
            ]
        )
        self.route_scale = nn.ParameterList(
            [
                nn.Parameter(torch.tensor(0.01))
                for _ in range(len(self.route_attention))
            ]
        )
        for budget, gate in zip(
            self.route_budget,
            self.route_gate,
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
        self.set_norm = (
            nn.LayerNorm(hidden_dim)
            if int(attention_layers) > 0
            else None
        )
        self.set_ffn = (
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            if int(attention_layers) > 0
            else None
        )
        self.global_query = (
            nn.Linear(hidden_dim * 2, hidden_dim)
            if int(attention_layers) > 0
            else None
        )
        if self.global_query is not None:
            nn.init.normal_(
                self.global_query.weight,
                mean=0.0,
                std=1.0e-3,
            )
            nn.init.zeros_(self.global_query.bias)
        for module in (self.presence[-1], self.count[-1]):
            nn.init.normal_(module.weight, mean=0.0, std=1.0e-3)
            nn.init.zeros_(module.bias)
        nn.init.normal_(self.class_count.weight, mean=0.0, std=1.0e-3)
        if class_count_prior is None:
            nn.init.zeros_(self.class_count.bias)
            self.register_buffer(
                "class_count_prior",
                torch.empty(0),
                persistent=False,
            )
        else:
            self.register_buffer(
                "class_count_prior",
                class_count_prior.detach().float().flatten(),
                persistent=False,
            )
            self.reset_class_count_bias()
        count_values = torch.arange(
            self.num_queries + 1,
            dtype=self.count[-1].bias.dtype,
        )
        normal = -0.5 * (
            (count_values - min(8, self.num_queries)) / 3.0
        ).square()
        with torch.no_grad():
            self.count[-1].bias.copy_(normal.log_softmax(dim=0))

    def reset_class_count_bias(self) -> None:
        """Reapply the configured sparse prior after checkpoint loading."""

        if self.class_count_prior.numel() == 0:
            return
        with torch.no_grad():
            self.class_count.bias.copy_(
                _class_count_log_prior(
                    self.class_count_prior,
                    self.num_queries,
                )
            )

    def forward(
        self,
        features: torch.Tensor,
        query_features: torch.Tensor,
        mask_logits: torch.Tensor,
        class_existence_logits: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if features.ndim != 4 or query_features.ndim != 3:
            raise ValueError("features/query features must be [B,C,H,W]/[B,Q,D]")
        if mask_logits.ndim != 4:
            raise ValueError("mask logits must be [B,Q,H,W]")
        if mask_logits.shape[1] != self.num_queries:
            raise ValueError("mask query count does not match the set head")
        spatial_size = tuple(
            min(self.pool_size, dimension)
            for dimension in features.shape[-2:]
        )
        projected = F.adaptive_avg_pool2d(
            self.feature_projection(features.float()),
            spatial_size,
        )
        probability = F.adaptive_avg_pool2d(
            mask_logits.float().sigmoid(),
            spatial_size,
        )
        mass = probability.mean(dim=(2, 3))
        peak = probability.amax(dim=(2, 3))
        area = (probability >= 0.5).float().mean(dim=(2, 3))
        global_summary = torch.cat(
            [
                projected.mean(dim=(2, 3)),
                projected.amax(dim=(2, 3)),
            ],
            dim=-1,
        )
        set_tokens = (
            self.query_projection(query_features.float())
            if self.query_projection is not None
            else query_features.float()
        )
        if self.set_attention:
            set_tokens = set_tokens + self.global_query(global_summary)[
                :, None
            ]
            route_entropy = []
            route_attention_entropy = []
            route_expert_fraction = []
            route_budget_values = []
            route_gate_values = []
            for layer_index, layer in enumerate(self.set_attention):
                normalized = self.set_norm(set_tokens)
                dense_update = layer(
                    normalized,
                    normalized,
                    normalized,
                    need_weights=False,
                )[0]
                if self.routing_attention:
                    route = self.route_attention[layer_index]
                    pooled = torch.cat(
                        [
                            normalized.mean(dim=1),
                            normalized.amax(dim=1),
                        ],
                        dim=-1,
                    )
                    router_count = min(
                        self.route_router_count,
                        normalized.shape[1],
                    )
                    base_top_k = min(
                        self.route_top_k,
                        normalized.shape[1],
                    )
                    budget_soft = 1.0 + torch.sigmoid(
                        self.route_budget[layer_index](pooled)
                    )[..., :router_count] * (
                        base_top_k - 1
                    )
                    budgets = budget_soft.detach().round().clamp(
                        1,
                        base_top_k,
                    ).to(torch.long)
                    routed, diagnostics = route(
                        normalized,
                        normalized,
                        router_budgets=budgets,
                    )
                    gate = torch.sigmoid(
                        self.route_gate[layer_index](
                            torch.cat([normalized, routed], dim=-1)
                        )
                    )
                    budget_strength = (
                        budget_soft.mean(dim=1)
                        / base_top_k
                    )[:, None, None]
                    dense_update = dense_update + (
                        torch.tanh(self.route_scale[layer_index])
                        * budget_strength
                        * gate
                        * routed
                    )
                    route_entropy.append(diagnostics["routing_entropy"])
                    route_attention_entropy.append(
                        diagnostics["attention_entropy"]
                    )
                    route_expert_fraction.append(
                        diagnostics["effective_expert_fraction"]
                    )
                    route_budget_values.append(budget_soft.mean())
                    route_gate_values.append(gate.mean())
                set_tokens = set_tokens + dense_update
            set_tokens = set_tokens + self.set_ffn(set_tokens)
        else:
            route_entropy = []
            route_attention_entropy = []
            route_expert_fraction = []
            route_budget_values = []
            route_gate_values = []
        presence_input = torch.cat(
            [
                set_tokens,
                torch.stack([mass, peak, area], dim=-1),
            ],
            dim=-1,
        )
        query_presence = (
            class_existence_logits
            + self.presence(presence_input).squeeze(-1)
        )
        set_summary = torch.stack(
            [
                query_presence.sigmoid().sum(dim=1)
                / max(1, self.num_queries),
                mass.sum(dim=1) / max(1, self.num_queries),
                peak.mean(dim=1),
            ],
            dim=-1,
        )
        learned_count_logits = self.count(
            torch.cat([global_summary, set_summary], dim=-1)
        )
        presence_count_logits = poisson_binomial_log_probability(
            query_presence
        )
        count_logits = (
            learned_count_logits
            + torch.tanh(self.presence_count_scale) * presence_count_logits
        )
        selection_result = query_count_selection_from_presence(
            query_presence,
            count_logits,
        )
        class_count_logits = self.class_count(global_summary).reshape(
            features.shape[0],
            self.num_room_classes,
            self.num_queries + 1,
        )
        diagnostics = {
            "presence_mean": query_presence.sigmoid().mean(),
            "count_entropy": _entropy(count_logits).mean(),
            "presence_count_scale": torch.tanh(
                self.presence_count_scale
            ),
            "presence_expected_count": (
                count_logits.softmax(dim=-1)
                * torch.arange(
                    count_logits.shape[-1],
                    device=count_logits.device,
                    dtype=count_logits.dtype,
                )
            ).sum(dim=-1).mean(),
        }
        if route_entropy:
            diagnostics.update(
                {
                    "route_entropy": torch.stack(route_entropy).mean(),
                    "route_attention_entropy": torch.stack(
                        route_attention_entropy
                    ).mean(),
                    "route_effective_expert_fraction": torch.stack(
                        route_expert_fraction
                    ).mean(),
                    "route_budget_mean": torch.stack(
                        route_budget_values
                    ).mean(),
                    "route_gate_mean": torch.stack(route_gate_values).mean(),
                }
            )
        return {
            "query_existence_logits": query_presence,
            "room_count_logits": count_logits,
            "room_presence_count_logits": presence_count_logits,
            "query_count_candidate_mask": selection_result["selection"],
            "query_count_soft_selection": selection_result["soft_selection"],
            "room_class_count_logits": class_count_logits,
            "query_set_diagnostics": diagnostics,
        }


def query_count_selection_from_presence(
    query_presence: torch.Tensor,
    count_logits: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Select one slot ranking with hard and differentiable count gates."""

    if query_presence.ndim != 2:
        raise ValueError("query presence must be [B,Q]")
    queries = query_presence.shape[1]
    if count_logits.shape != (query_presence.shape[0], queries + 1):
        raise ValueError("count logits must be [B,Q+1]")
    count = count_logits.argmax(dim=-1)
    indices = query_presence.topk(queries, dim=1).indices
    positions = torch.arange(
        queries,
        device=count_logits.device,
    )
    selection = torch.zeros(
        query_presence.shape,
        dtype=torch.bool,
        device=query_presence.device,
    )
    selection.scatter_(
        1,
        indices,
        positions[None, :] < count[:, None],
    )
    count_probability = count_logits.float().softmax(dim=-1)
    count_cdf = count_probability.cumsum(dim=-1)[..., :queries]
    soft_ordered = (1.0 - count_cdf).to(query_presence.dtype)
    soft_selection = torch.zeros_like(soft_ordered)
    soft_selection.scatter_(1, indices, soft_ordered)
    return {
        "selection": selection,
        "soft_selection": soft_selection,
        "ordered_indices": indices,
        "count": count,
    }


def _entropy(logits: torch.Tensor) -> torch.Tensor:
    probability = logits.float().softmax(dim=-1)
    return -(
        probability.clamp_min(1.0e-8)
        * probability.clamp_min(1.0e-8).log()
    ).sum(dim=-1)


def _class_count_log_prior(
    class_prior: torch.Tensor,
    num_queries: int,
) -> torch.Tensor:
    """Allocate the initial room budget across classes without data leakage."""

    prior = class_prior.detach().float().flatten()
    if prior.numel() == 0:
        raise ValueError("class-count prior must be nonempty")
    if prior.numel() == 0 or float(prior.sum()) <= 0.0:
        prior = torch.ones_like(prior)
    expected_total = float(min(8, max(1, int(num_queries))))
    expected = expected_total * prior.clamp_min(0.0) / prior.sum().clamp_min(1.0e-8)
    counts = torch.arange(
        int(num_queries) + 1,
        dtype=expected.dtype,
        device=expected.device,
    )
    factorial = torch.lgamma(counts + 1.0)
    log_probability = (
        counts[None] * expected[:, None].clamp_min(1.0e-8).log()
        - expected[:, None]
        - factorial[None]
    )
    return log_probability.reshape(-1)


__all__ = [
    "RoomQuerySetHead",
    "_one_to_one_query_targets",
    "query_count_selection_from_presence",
    "room_query_set_loss",
]
