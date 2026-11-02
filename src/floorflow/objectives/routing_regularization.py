"""Regularizers that keep typed routing informative instead of collapsed."""

from __future__ import annotations

import math

import torch


def _entropy_floor(value: torch.Tensor | None, floor: float) -> torch.Tensor:
    if value is None:
        return torch.zeros((), device="cpu")
    return (float(floor) - value.float()).clamp_min(0.0)


def typed_routing_regularization(
    model: torch.nn.Module,
    output: dict[str, torch.Tensor],
    *,
    source_entropy_floor: float = 0.45,
    source_usage_entropy_floor: float = 0.45,
    channel_entropy_floor: float = 0.30,
    stage_entropy_floor: float = 0.40,
    expert_fraction_floor: float = 0.15,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Penalize collapsed source, channel, and CAD-stage routing."""

    source = output.get("field_source_routing")
    if isinstance(source, dict):
        device = source["source_weights"].device
        source_entropy = source.get("source_entropy")
        usage = source["source_weights"].mean(dim=(0, 3, 4))
        usage_entropy = -(
            usage.clamp_min(1.0e-8) * usage.clamp_min(1.0e-8).log()
        ).sum(dim=-1)
        usage_entropy = usage_entropy / max(
            1.0,
            math.log(max(2, usage.shape[-1])),
        )
        source_loss = _entropy_floor(
            source_entropy,
            source_entropy_floor,
        ).to(device)
        usage_loss = (
            float(source_usage_entropy_floor) - usage_entropy
        ).clamp_min(0.0).mean()
    else:
        device = next(model.parameters()).device
        source_loss = torch.zeros((), device=device)
        usage_loss = torch.zeros((), device=device)

    channel_loss = _entropy_floor(
        output.get("channel_source_attention_entropy"),
        channel_entropy_floor,
    ).to(device)
    stage_losses = []
    diagnostics = output.get("action_route_diagnostics")
    if isinstance(diagnostics, dict):
        for values in diagnostics.values():
            if not isinstance(values, dict):
                continue
            stage_losses.append(
                _entropy_floor(
                    values.get("routing_entropy"),
                    stage_entropy_floor,
                ).to(device)
            )
            stage_losses.append(
                _entropy_floor(
                    values.get("effective_expert_fraction"),
                    expert_fraction_floor,
                ).to(device)
            )
    query_diagnostics = output.get("query_set_diagnostics")
    if isinstance(query_diagnostics, dict):
        for key in ("route_entropy", "route_attention_entropy"):
            stage_losses.append(
                _entropy_floor(
                    query_diagnostics.get(key),
                    stage_entropy_floor,
                ).to(device)
            )
        stage_losses.append(
            _entropy_floor(
                query_diagnostics.get("route_effective_expert_fraction"),
                expert_fraction_floor,
            ).to(device)
        )
    stage_loss = (
        torch.stack(stage_losses).mean()
        if stage_losses
        else torch.zeros((), device=device)
    )
    rank_losses = [
        module.rank_budget_loss().to(device)
        for module in model.modules()
        if hasattr(module, "rank_budget_loss")
    ]
    rank_loss = (
        torch.stack(rank_losses).mean()
        if rank_losses
        else torch.zeros((), device=device)
    )
    total = (
        source_loss
        + usage_loss
        + channel_loss
        + stage_loss
        + 0.1 * rank_loss
    )
    return total, {
        "routing_source_floor_loss": source_loss.detach(),
        "routing_source_usage_floor_loss": usage_loss.detach(),
        "routing_channel_floor_loss": channel_loss.detach(),
        "routing_stage_floor_loss": stage_loss.detach(),
        "routing_rank_budget_loss": rank_loss.detach(),
        "routing_regularization_loss": total.detach(),
    }


__all__ = ["typed_routing_regularization"]
