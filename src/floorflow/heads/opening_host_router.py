"""Opening-query routing to any selected wall and 1D host interval."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.geometry.wall_graph_readout import ReadoutOpening
from floorflow.heads.attention_masks import unmask_first_empty_row
from floorflow.model.official_mita_route import OfficialMiTARouteAttention


class OpeningHostRouter(nn.Module):
    """Predict one wall host per opening query without per-wall exclusivity."""

    def __init__(
        self,
        hidden_dim: int,
        num_types: int = 2,
        heads: int = 4,
        route_attention: bool = False,
        route_router_count: int = 16,
        route_top_k: int = 8,
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim)
        if hidden_dim % heads:
            heads = 1
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim,
            heads,
            batch_first=True,
        )
        self.opening_norm = nn.LayerNorm(hidden_dim)
        self.wall_norm = nn.LayerNorm(hidden_dim)
        self.opening_projection = nn.Linear(hidden_dim, hidden_dim)
        self.wall_projection = nn.Linear(hidden_dim, hidden_dim)
        self.host_bias = nn.Linear(hidden_dim, 1)
        self.interval = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )
        self.presence = nn.Linear(hidden_dim, 1)
        self.type_head = nn.Linear(hidden_dim, int(num_types))
        self.route_attention = bool(route_attention)
        self.route_router_count = max(1, int(route_router_count))
        self.route_top_k = max(1, int(route_top_k))
        self.route = (
            OfficialMiTARouteAttention(
                hidden_dim,
                num_heads=heads,
                router_count=self.route_router_count,
                top_k=self.route_top_k,
            )
            if self.route_attention
            else None
        )
        self.route_budget = (
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, self.route_router_count),
            )
            if self.route_attention
            else None
        )
        self.route_gate = (
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )
            if self.route_attention
            else None
        )
        self.route_scale = (
            nn.Parameter(torch.tensor(0.01))
            if self.route_attention
            else None
        )
        if self.route_budget is not None and self.route_gate is not None:
            nn.init.zeros_(self.route_budget[-1].weight)
            nn.init.zeros_(self.route_budget[-1].bias)
            nn.init.zeros_(self.route_gate[-1].weight)
            nn.init.zeros_(self.route_gate[-1].bias)
        nn.init.zeros_(self.interval[-1].weight)
        nn.init.zeros_(self.interval[-1].bias)
        nn.init.zeros_(self.host_bias.weight)
        nn.init.zeros_(self.host_bias.bias)

    def forward(
        self,
        wall_tokens: torch.Tensor,
        opening_tokens: torch.Tensor,
        wall_endpoints: torch.Tensor,
        *,
        opening_centers: torch.Tensor | None = None,
        opening_widths: torch.Tensor | None = None,
        wall_mask: torch.Tensor | None = None,
        opening_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if wall_tokens.ndim != 3 or opening_tokens.ndim != 3:
            raise ValueError("wall and opening tokens must be [B,N,D]")
        if wall_endpoints.shape != (
            wall_tokens.shape[0],
            wall_tokens.shape[1],
            2,
            2,
        ):
            raise ValueError("wall_endpoints must be [B,W,2,2]")
        wall_valid = (
            torch.ones(wall_tokens.shape[:2], dtype=torch.bool, device=wall_tokens.device)
            if wall_mask is None
            else wall_mask.bool()
        )
        opening_valid = (
            torch.ones(
                opening_tokens.shape[:2],
                dtype=torch.bool,
                device=opening_tokens.device,
            )
            if opening_mask is None
            else opening_mask.bool()
        )
        empty = opening_tokens.shape[1] == 0 or wall_tokens.shape[1] == 0
        route_diagnostics = None
        if empty:
            attended = torch.zeros_like(opening_tokens)
        else:
            key_padding = ~wall_valid
            key_padding = unmask_first_empty_row(key_padding)
            attended = self.cross_attention(
                self.opening_norm(opening_tokens),
                self.wall_norm(wall_tokens),
                self.wall_norm(wall_tokens),
                key_padding_mask=key_padding,
                need_weights=False,
            )[0]
            if (
                self.route is not None
                and self.route_budget is not None
                and self.route_gate is not None
                and self.route_scale is not None
            ):
                route_wall_mask = ~key_padding
                wall_summary = (
                    wall_tokens
                    * route_wall_mask[..., None].to(wall_tokens.dtype)
                ).sum(dim=1) / route_wall_mask.sum(
                    dim=1,
                    keepdim=True,
                ).clamp_min(1)
                router_count = min(
                    self.route_router_count,
                    opening_tokens.shape[1],
                )
                base_top_k = min(
                    self.route_top_k,
                    wall_tokens.shape[1],
                )
                budget_soft = 1.0 + torch.sigmoid(
                    self.route_budget(
                        torch.cat(
                            [
                                opening_tokens.mean(dim=1),
                                wall_summary,
                            ],
                            dim=-1,
                        )
                    )
                )[..., :router_count] * (base_top_k - 1)
                budgets = budget_soft.detach().round().clamp(
                    1,
                    base_top_k,
                ).to(torch.long)
                routed, route_diagnostics = self.route(
                    self.opening_norm(opening_tokens),
                    self.wall_norm(wall_tokens),
                    key_value_mask=route_wall_mask,
                    router_budgets=budgets,
                )
                route_gate = torch.sigmoid(
                    self.route_gate(torch.cat([attended, routed], dim=-1))
                )
                budget_strength = (
                    budget_soft.mean(dim=1) / base_top_k
                )[:, None, None]
                attended = attended + (
                    torch.tanh(self.route_scale)
                    * budget_strength
                    * route_gate
                    * routed
                )
                route_diagnostics = {
                    **route_diagnostics,
                    "route_budget_soft_mean": budget_soft.mean(),
                    "route_gate_mean": route_gate.mean(),
                }
        opening_hidden = opening_tokens + attended
        opening_projected = self.opening_projection(opening_hidden)
        wall_projected = self.wall_projection(wall_tokens)
        learned_host_logits = torch.einsum(
            "bod,bwd->bow",
            opening_projected,
            wall_projected,
        ) / math.sqrt(float(wall_tokens.shape[-1]))
        learned_host_logits = (
            learned_host_logits + self.host_bias(opening_hidden)
        )
        if opening_centers is not None:
            starts = wall_endpoints[:, :, 0]
            ends = wall_endpoints[:, :, 1]
            delta = ends - starts
            length_sq = (delta**2).sum(dim=-1).clamp_min(1.0e-9)
            relative = opening_centers[:, :, None] - starts[:, None]
            fraction = (
                (relative * delta[:, None]).sum(dim=-1)
                / length_sq[:, None]
            ).clamp(0.0, 1.0)
            projected = starts[:, None] + fraction[..., None] * delta[:, None]
            distance = (opening_centers[:, :, None] - projected).norm(dim=-1)
            host_logits = learned_host_logits - distance / 0.02
        else:
            host_logits = learned_host_logits
        host_logits = host_logits.masked_fill(
            ~wall_valid[:, None, :],
            -1.0e4,
        )
        raw_interval = self.interval(opening_hidden)
        center_residual = 0.2 * torch.tanh(raw_interval[..., 0])
        width_residual = 0.2 * torch.tanh(raw_interval[..., 1])
        if wall_tokens.shape[1]:
            host_index = host_logits.argmax(dim=-1)
            if opening_centers is not None:
                starts = wall_endpoints[:, :, 0]
                ends = wall_endpoints[:, :, 1]
                delta = ends - starts
                length_sq = (delta**2).sum(dim=-1).clamp_min(1.0e-9)
                relative = opening_centers[:, :, None] - starts[:, None]
                host_fraction = (
                    (relative * delta[:, None]).sum(dim=-1)
                    / length_sq[:, None]
                ).clamp(0.0, 1.0)
                prior_center = torch.gather(
                    host_fraction,
                    2,
                    host_index[..., None],
                ).squeeze(-1)
                center_fraction = (
                    prior_center + center_residual
                ).clamp(0.0, 1.0)
                if opening_widths is not None:
                    host_length = delta.norm(dim=-1).clamp_min(1.0e-6)
                    width_prior = (
                        opening_widths[:, :, None] / host_length[:, None]
                    ).clamp(0.0, 1.0)
                    prior_width = torch.gather(
                        width_prior,
                        2,
                        host_index[..., None],
                    ).squeeze(-1)
                    width_fraction = (
                        prior_width + width_residual
                    ).clamp(0.0, 1.0)
                else:
                    width_fraction = torch.sigmoid(
                        raw_interval[..., 1]
                    )
            else:
                center_fraction = torch.sigmoid(raw_interval[..., 0])
                width_fraction = torch.sigmoid(raw_interval[..., 1])
            gather = host_index[..., None, None, None].expand(
                -1,
                -1,
                1,
                2,
                2,
            )
            host_endpoints = torch.gather(
                wall_endpoints[:, None].expand(
                    -1,
                    opening_tokens.shape[1],
                    -1,
                    -1,
                    -1,
                ),
                2,
                gather,
            ).squeeze(2)
        else:
            center_fraction = torch.sigmoid(raw_interval[..., 0])
            width_fraction = torch.sigmoid(raw_interval[..., 1])
            host_index = torch.zeros(
                opening_tokens.shape[:2],
                dtype=torch.long,
                device=opening_tokens.device,
            )
            host_endpoints = torch.zeros(
                opening_tokens.shape[0],
                opening_tokens.shape[1],
                2,
                2,
                device=opening_tokens.device,
            )
        center_xy = (
            host_endpoints[:, :, 0]
            + center_fraction[..., None]
            * (host_endpoints[:, :, 1] - host_endpoints[:, :, 0])
        )
        return {
            "host_logits": host_logits,
            "host_index": host_index,
            "center_fraction": center_fraction,
            "width_fraction": width_fraction,
            "center_xy": center_xy,
            "host_endpoints": host_endpoints,
            "presence_logits": self.presence(opening_hidden).squeeze(-1),
            "type_logits": self.type_head(opening_hidden),
            "wall_mask": wall_valid,
            "opening_mask": opening_valid,
            "route_diagnostics": route_diagnostics,
        }


def opening_route_metrics(
    predictions: dict[str, object],
) -> dict[str, float]:
    """Flatten MIta route health for opening-router validation."""

    diagnostics = predictions.get("route_diagnostics")
    if not isinstance(diagnostics, dict):
        return {}
    values = {
        "route_budget": diagnostics.get("route_budget_soft_mean"),
        "route_entropy": diagnostics.get("routing_entropy"),
        "route_attention_entropy": diagnostics.get("attention_entropy"),
        "route_effective_expert_fraction": diagnostics.get(
            "effective_expert_fraction"
        ),
        "route_gate_mean": diagnostics.get("route_gate_mean"),
    }
    return {
        name: float(value.detach().float().mean().cpu())
        for name, value in values.items()
        if isinstance(value, torch.Tensor)
        and torch.isfinite(value.detach().float().mean())
    }


def opening_host_router_loss(
    predictions: dict[str, torch.Tensor],
    target_hosts: torch.Tensor,
    target_center_fraction: torch.Tensor,
    target_width_fraction: torch.Tensor,
    target_types: torch.Tensor,
    matched_mask: torch.Tensor,
    *,
    presence_weight: float = 0.5,
    host_weight: float = 1.0,
    interval_weight: float = 1.0,
    type_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    selected = matched_mask.bool()
    opening_valid = predictions["opening_mask"].bool()
    presence_target = selected.to(predictions["presence_logits"].dtype)
    presence_loss = (
        F.binary_cross_entropy_with_logits(
            predictions["presence_logits"][opening_valid],
            presence_target[opening_valid],
        )
        if opening_valid.any()
        else predictions["presence_logits"].sum() * 0.0
    )
    if selected.any():
        host_loss = F.cross_entropy(
            predictions["host_logits"][selected],
            target_hosts[selected],
        )
        center_loss = F.l1_loss(
            predictions["center_fraction"][selected],
            target_center_fraction[selected],
        )
        width_loss = F.l1_loss(
            predictions["width_fraction"][selected],
            target_width_fraction[selected],
        )
        type_loss = F.cross_entropy(
            predictions["type_logits"][selected],
            target_types[selected],
        )
    else:
        zero = predictions["host_logits"].sum() * 0.0
        host_loss = zero
        center_loss = zero
        width_loss = zero
        type_loss = zero
    loss = (
        float(presence_weight) * presence_loss
        + float(host_weight) * host_loss
        + float(interval_weight) * (center_loss + width_loss)
        + float(type_weight) * type_loss
    )
    return loss, {
        "opening_query_presence_loss": presence_loss.detach(),
        "opening_host_loss": host_loss.detach(),
        "opening_center_fraction_loss": center_loss.detach(),
        "opening_width_fraction_loss": width_loss.detach(),
        "opening_type_loss": type_loss.detach(),
    }


def slice_tensor_predictions(
    predictions: dict[str, object],
    sample_index: int,
) -> dict[str, torch.Tensor]:
    """Clone one batch row while dropping non-tensor routing diagnostics."""

    return {
        key: value[sample_index : sample_index + 1].clone()
        for key, value in predictions.items()
        if torch.is_tensor(value) and value.ndim > 0
    }


def routed_opening_rows(
    predictions: dict[str, torch.Tensor],
    sample_index: int,
    taxonomy: FloorplanTaxonomy,
    *,
    frame: float = 1.0,
    threshold: float | None = None,
    query_centers: torch.Tensor | None = None,
    query_widths: torch.Tensor | None = None,
    query_families: torch.Tensor | None = None,
    preserve_query_intervals: bool = False,
) -> list[ReadoutOpening]:
    """Convert routed query predictions into the CAD opening layer."""

    rows = []
    presence = predictions["presence_logits"][sample_index].sigmoid()
    active = predictions["opening_mask"][sample_index]
    if threshold is not None:
        active = active & (presence >= float(threshold))
    for index in torch.nonzero(
        active,
        as_tuple=False,
    ).flatten().tolist():
        host = int(predictions["host_index"][sample_index, index].cpu())
        host_end = predictions["host_endpoints"][sample_index, index]
        host_length = (
            float((host_end[1] - host_end[0]).norm().cpu()) * float(frame)
        )
        if host_length <= 1.0e-6:
            continue
        if query_families is not None:
            kind = (
                "window"
                if int(query_families[sample_index, index].cpu()) == 1
                else "door"
            )
        else:
            class_index = int(
                predictions["type_logits"][sample_index, index].argmax().cpu()
            )
            opening_class = int(taxonomy.opening_ids[class_index])
            kind = (
                "window"
                if taxonomy.class_names[opening_class].startswith("window:")
                else "door"
            )
        center = tuple(
            float(value.detach()) * float(frame)
            for value in predictions["center_xy"][sample_index, index].cpu()
        )
        width = max(
            1.0,
            float(
                predictions["width_fraction"][
                    sample_index,
                    index,
                ].detach().cpu()
            )
            * host_length,
        )
        if preserve_query_intervals and query_centers is not None:
            source = query_centers[sample_index, index].detach().cpu()
            start = host_end[0].detach().cpu()
            end = host_end[1].detach().cpu()
            delta = end - start
            fraction = float(
                ((source - start) * delta).sum().clamp(0.0, 1.0)
                / (delta**2).sum().clamp_min(1.0e-9)
            )
            projected = (start + fraction * delta) * float(frame)
            center = (float(projected[0]), float(projected[1]))
        if preserve_query_intervals and query_widths is not None:
            width = min(
                host_length,
                max(
                    1.0,
                    float(query_widths[sample_index, index].detach().cpu())
                    * float(frame),
                ),
            )
        rows.append(
            ReadoutOpening(
                center=center,
                width=width,
                kind=kind,
                wall_index=host,
            )
        )
    return rows


__all__ = [
    "OpeningHostRouter",
    "opening_host_router_loss",
    "opening_route_metrics",
    "routed_opening_rows",
    "slice_tensor_predictions",
]
