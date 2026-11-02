"""TopoGED-style multiscale descriptors for CAD wall candidates."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def endpoint_adjacency(
    endpoints: torch.Tensor,
    valid: torch.Tensor,
    *,
    tolerance: float = 0.01,
) -> torch.Tensor:
    """Connect candidates that share a junction within frame tolerance."""

    starts = endpoints[:, :, 0]
    ends = endpoints[:, :, 1]
    distances = torch.minimum(
        torch.cdist(starts, starts),
        torch.cdist(starts, ends),
    )
    distances = torch.minimum(distances, torch.cdist(ends, starts))
    distances = torch.minimum(distances, torch.cdist(ends, ends))
    adjacency = distances <= float(tolerance)
    adjacency &= valid[:, :, None] & valid[:, None, :]
    adjacency &= ~torch.eye(
        endpoints.shape[1],
        dtype=torch.bool,
        device=endpoints.device,
    )[None]
    return adjacency.to(endpoints.dtype)


def filtration_descriptor(
    endpoints: torch.Tensor,
    weights: torch.Tensor,
    valid: torch.Tensor,
    *,
    thresholds: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0),
    tolerance: float = 0.01,
    temperature: float = 0.15,
) -> torch.Tensor:
    """Count nodes and induced links at cumulative junction degrees."""

    adjacency = endpoint_adjacency(
        endpoints,
        valid,
        tolerance=tolerance,
    )
    degree = adjacency @ weights[..., None]
    node_rows = []
    edge_rows = []
    pair_adjacency = torch.triu(
        adjacency * weights[:, :, None] * weights[:, None, :],
        diagonal=1,
    )
    for threshold in thresholds:
        node_gate = torch.sigmoid(
            (float(threshold) - degree[..., 0]) / float(temperature)
        )
        node_gate = node_gate * valid.to(node_gate.dtype)
        node_rows.append((weights * node_gate).sum(dim=1))
        induced_edges = pair_adjacency * node_gate[:, :, None] * node_gate[:, None, :]
        edge_rows.append(induced_edges.sum(dim=(1, 2)))
    return torch.stack([*node_rows, *edge_rows], dim=1)


def graphlet_descriptor_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """MSE, histogram-intersection, and cumulative L1 descriptor loss."""

    predicted_dist = predicted / (
        predicted.sum(dim=1, keepdim=True) + eps
    )
    target_dist = target / (target.sum(dim=1, keepdim=True) + eps)
    mse = F.mse_loss(predicted_dist, target_dist)
    intersection = torch.minimum(predicted_dist, target_dist).sum(dim=1)
    histogram = 1.0 - intersection.mean()
    cumulative = F.l1_loss(
        predicted_dist.cumsum(dim=1),
        target_dist.cumsum(dim=1),
    )
    return mse + histogram + cumulative


__all__ = [
    "endpoint_adjacency",
    "filtration_descriptor",
    "graphlet_descriptor_loss",
]
