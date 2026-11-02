"""Soft graph edit distance used only as a training signal."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from floorflow.objectives.token_optimal_transport import sinkhorn_plan


def soft_typed_matching_cost(
    predicted_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    predicted_classes: torch.Tensor,
    target_classes: torch.Tensor,
    *,
    class_weight: float = 1.0,
    degree_weight: float = 0.0,
    geometry_weight: float = 0.0,
) -> torch.Tensor:
    """Build a typed node cost without changing semantic room order."""

    geometry = torch.cdist(predicted_tokens.float(), target_tokens.float()).pow(2)
    class_cost = 1.0 - torch.einsum(
        "bpc,btc->bpt",
        predicted_classes.float().softmax(dim=-1),
        target_classes.float().softmax(dim=-1),
    )
    predicted_degree = predicted_tokens.new_tensor(0.0)
    if predicted_tokens.shape[-1] >= 2:
        predicted_degree = predicted_tokens[..., -1]
        target_degree = target_tokens[..., -1]
        degree_cost = (
            predicted_degree[:, :, None] - target_degree[:, None, :]
        ).abs()
    else:
        degree_cost = torch.zeros_like(predicted_degree)
    return (
        float(geometry_weight) * geometry
        + float(class_weight) * class_cost
        + float(degree_weight) * degree_cost
    )


def soft_graph_edit_distance(
    predicted_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    predicted_classes: torch.Tensor,
    target_classes: torch.Tensor,
    predicted_adjacency: torch.Tensor,
    target_adjacency: torch.Tensor,
    *,
    predicted_valid: torch.Tensor | None = None,
    target_valid: torch.Tensor | None = None,
    epsilon: float = 0.1,
    iterations: int = 20,
    existence_weight: float = 2.0,
    geometry_weight: float = 0.05,
    aligned_node_order: bool = True,
    aligned_edge_order: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Sinkhorn node transport plus soft edge disagreement."""

    cost = soft_typed_matching_cost(
        predicted_tokens,
        target_tokens,
        predicted_classes,
        target_classes,
        geometry_weight=geometry_weight,
    )
    target_mask = (
        torch.ones(
            target_tokens.shape[:2],
            device=cost.device,
            dtype=torch.bool,
        )
        if target_valid is None
        else target_valid.bool()
    )
    predicted_mask = (
        torch.ones(
            predicted_tokens.shape[:2],
            device=cost.device,
            dtype=torch.bool,
        )
        if predicted_valid is None
        else predicted_valid.bool()
    )
    invalid_penalty = cost.amax(dim=(1, 2), keepdim=True).detach() + 1.0
    cost = torch.where(target_mask[:, None, :], cost, invalid_penalty.expand_as(cost))
    plan = sinkhorn_plan(cost, epsilon=epsilon, iterations=iterations)
    plan = plan * predicted_mask[:, :, None].float()
    plan = plan / plan.sum(dim=(1, 2), keepdim=True).clamp_min(1.0e-6)
    node_loss = (plan * cost).sum(dim=(1, 2)).mean()
    if aligned_node_order:
        transported_presence = predicted_mask.float()
    else:
        transported_presence = torch.bmm(
            plan.transpose(1, 2),
            predicted_mask[:, :, None].float(),
        )[:, :, 0]
    existence_loss = F.binary_cross_entropy_with_logits(
        torch.logit(
            transported_presence.clamp(1.0e-5, 1.0 - 1.0e-5)
        )[target_mask],
        target_mask.float()[target_mask],
    )
    if aligned_edge_order:
        transported_edges = predicted_adjacency.float()
    else:
        transported_edges = torch.bmm(
            torch.bmm(plan.transpose(1, 2), predicted_adjacency.float()),
            plan,
        )
    transported_edges = transported_edges / transported_edges.amax(
        dim=(1, 2),
        keepdim=True,
    ).clamp_min(1.0e-6)
    edge_mask = target_mask[:, :, None] & target_mask[:, None, :]
    target_count = target_mask.sum(dim=1).clamp_min(1).float()
    edge_loss = (
        (transported_edges - target_adjacency.float()).abs() * edge_mask
    ).sum(dim=(1, 2)).div(target_count).mean()
    node_term = (
        node_loss / target_count.mean()
        if not aligned_node_order
        else cost.sum() * 0.0
    )
    total = (
        node_term
        + float(existence_weight) * existence_loss
        + edge_loss
    )
    return total, {
        "soft_node_edit": node_term.detach(),
        "soft_existence_edit": existence_loss.detach(),
        "soft_edge_edit": edge_loss.detach(),
        "soft_transport_entropy": (
            -(plan * plan.clamp_min(1.0e-8).log()).sum(dim=(1, 2)).mean().detach()
        ),
    }


__all__ = ["soft_graph_edit_distance", "soft_typed_matching_cost"]
