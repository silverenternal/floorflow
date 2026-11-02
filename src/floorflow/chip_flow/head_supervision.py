"""Masked supervision contracts for the coupled chip hierarchy heads."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class HeadSupervisionTerms:
    total: torch.Tensor
    block: torch.Tensor
    block_center: torch.Tensor
    block_rho: torch.Tensor
    block_forced: torch.Tensor
    global_state: torch.Tensor
    structural: torch.Tensor
    port: torch.Tensor
    port_connected: torch.Tensor
    port_unconnected: torch.Tensor

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {
            "flow_block": self.block,
            "flow_block_structured": self.block,
            "flow_block_center": self.block_center,
            "flow_block_rho": self.block_rho,
            "flow_block_forced": self.block_forced,
            "flow_global": self.global_state,
            "flow_structural": self.structural,
            "flow_port": self.port,
            "flow_port_connected": self.port_connected,
            "flow_port_unconnected": self.port_unconnected,
        }


def smooth_l1_masked(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    beta: float,
) -> torch.Tensor:
    """Return a masked Smooth-L1 mean with a finite zero fallback."""

    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes must match")
    weight = mask.to(prediction.dtype)
    while weight.ndim < prediction.ndim:
        weight = weight.unsqueeze(-1)
    weight = weight.expand_as(prediction)
    count = weight.sum()
    error = F.smooth_l1_loss(
        prediction.float(),
        target.float(),
        beta=float(beta),
        reduction="none",
    )
    return (error * weight).sum() / count.clamp_min(1.0)


def block_supervision_masks(
    constraints: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return masks for free center coordinates and non-anchored rho."""

    preplaced = constraints[..., 1].ne(0)
    anchored = constraints[..., 0].ne(0) | preplaced
    center = (~preplaced)[..., None].expand(-1, -1, 2)
    rho = (~anchored)[..., None]
    return center, rho


def port_semantic_masks(
    incidence: torch.Tensor,
    port_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split valid pins into block-connected and unconnected supervision."""

    connected = incidence.to(torch.bool).any(dim=1) & port_mask
    unconnected = (~incidence.to(torch.bool).any(dim=1)) & port_mask
    return connected, unconnected


def structured_hierarchy_flow_loss(
    *,
    block_prediction: torch.Tensor,
    block_target: torch.Tensor,
    global_prediction: torch.Tensor,
    global_target: torch.Tensor,
    structural_prediction: torch.Tensor,
    structural_target: torch.Tensor,
    port_prediction: torch.Tensor,
    port_target: torch.Tensor,
    constraints: torch.Tensor,
    structural_mask: torch.Tensor,
    port_incidence: torch.Tensor,
    port_mask: torch.Tensor,
    huber_beta: float,
    global_weight: float,
    structural_weight: float,
    port_weight: float,
    forced_weight: float = 0.0,
) -> HeadSupervisionTerms:
    """Compute level-normalized and semantically masked hierarchy losses."""

    center_mask, rho_mask = block_supervision_masks(constraints)
    block_center = smooth_l1_masked(
        block_prediction[..., :2],
        block_target[..., :2],
        center_mask,
        beta=huber_beta,
    )
    block_rho = smooth_l1_masked(
        block_prediction[..., 2:3],
        block_target[..., 2:3],
        rho_mask,
        beta=huber_beta,
    )
    preplaced = constraints[..., 1].ne(0)
    anchored = constraints[..., 0].ne(0) | preplaced
    forced_mask = torch.stack(
        [preplaced, preplaced, anchored],
        dim=-1,
    )
    forced = smooth_l1_masked(
        block_prediction,
        block_target,
        forced_mask,
        beta=huber_beta,
    )
    valid_center = center_mask.sum().clamp_min(1.0)
    valid_rho = rho_mask.sum().clamp_min(1.0)
    block = (
        block_center * valid_center / (valid_center + valid_rho)
        + block_rho * valid_rho / (valid_center + valid_rho)
    ) + float(forced_weight) * forced
    global_state = smooth_l1_masked(
        global_prediction,
        global_target,
        torch.ones_like(global_prediction, dtype=torch.bool),
        beta=huber_beta,
    )
    structural = smooth_l1_masked(
        structural_prediction,
        structural_target,
        structural_mask,
        beta=huber_beta,
    )
    connected_mask, unconnected_mask = port_semantic_masks(
        port_incidence,
        port_mask,
    )
    port_connected = smooth_l1_masked(
        port_prediction,
        port_target,
        connected_mask,
        beta=huber_beta,
    )
    port_unconnected = smooth_l1_masked(
        port_prediction,
        port_target,
        unconnected_mask,
        beta=huber_beta,
    )
    port_weights = torch.stack(
        [
            connected_mask.any().to(port_connected.dtype),
            unconnected_mask.any().to(port_unconnected.dtype),
        ]
    )
    port = (
        port_connected * port_weights[0]
        + port_unconnected * port_weights[1]
    ) / port_weights.sum().clamp_min(1.0)
    total = (
        block
        + float(global_weight) * global_state
        + float(structural_weight) * structural
        + float(port_weight) * port
    )
    return HeadSupervisionTerms(
        total=total,
        block=block,
        block_center=block_center,
        block_rho=block_rho,
        block_forced=forced,
        global_state=global_state,
        structural=structural,
        port=port,
        port_connected=port_connected,
        port_unconnected=port_unconnected,
    )


__all__ = [
    "HeadSupervisionTerms",
    "block_supervision_masks",
    "port_semantic_masks",
    "smooth_l1_masked",
    "structured_hierarchy_flow_loss",
]
