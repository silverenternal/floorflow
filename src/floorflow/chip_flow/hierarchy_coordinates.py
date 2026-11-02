"""Envelope and cluster-local coordinates for causal hierarchy coupling."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class HierarchyCoordinatePlan:
    """Per-block frame used to map global and local block states."""

    center: torch.Tensor
    scale: torch.Tensor
    envelope_center: torch.Tensor
    envelope_scale: torch.Tensor
    cluster_slot: torch.Tensor


class HierarchyBlockCoordinateGate(nn.Module):
    """Zero-initialized per-block strength for hierarchy coordinates."""

    def __init__(self, hidden: int, *, width: int | None = None) -> None:
        super().__init__()
        if hidden <= 0:
            raise ValueError("coordinate gate hidden size must be positive")
        bottleneck = int(width or max(8, hidden // 8))
        self.bias = nn.Parameter(torch.zeros(1))
        self.scorer = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, bottleneck),
            nn.SiLU(),
            nn.Linear(bottleneck, 1),
        )
        nn.init.zeros_(self.scorer[-1].weight)
        nn.init.zeros_(self.scorer[-1].bias)

    def forward(self, block_tokens: torch.Tensor) -> torch.Tensor:
        if block_tokens.ndim != 3:
            raise ValueError("coordinate gate requires block tokens")
        return (self.bias + self.scorer(block_tokens.float())).clamp(0.0, 1.0)


def build_hierarchy_coordinate_plan(
    block_state: torch.Tensor,
    global_state: torch.Tensor,
    structural_state: torch.Tensor,
    membership: torch.Tensor | None,
    structural_mask: torch.Tensor | None,
    *,
    cluster_local: bool = True,
) -> HierarchyCoordinatePlan:
    """Build target-free envelope or cluster frames from current hierarchy state."""

    if block_state.ndim != 3 or block_state.shape[-1] != 3:
        raise ValueError("block_state must have shape (batch, blocks, 3)")
    if global_state.shape != (block_state.shape[0], 4):
        raise ValueError("global_state must have shape (batch, 4)")
    center = block_state[..., :2].float()
    envelope_center = global_state[:, :2].float()
    envelope_scale = global_state[:, 2:4].float().abs().clamp_min(1.0e-3)
    frame_center = envelope_center[:, None].expand_as(center)
    frame_scale = envelope_scale[:, None].expand_as(center)
    cluster_slot = torch.full(
        block_state.shape[:2],
        -1,
        dtype=torch.long,
        device=block_state.device,
    )
    if cluster_local and membership is not None and structural_mask is not None:
        if membership.shape != (*block_state.shape[:2], structural_state.shape[1]):
            raise ValueError("membership shape must match blocks and structural slots")
        cluster_count = min(4, structural_state.shape[1])
        cluster_weights = (
            membership[..., :cluster_count].bool()
            & structural_mask[:, None, :cluster_count]
        )
        cluster_centers = structural_state[:, :cluster_count, :2].float()
        mass = cluster_weights.sum(dim=1)
        has_cluster = cluster_weights.any(dim=-1)
        cluster_slot = torch.where(
            has_cluster,
            cluster_weights.float().argmax(dim=-1),
            cluster_slot,
        )
        slot_delta = center[:, :, None] - cluster_centers[:, None]
        cluster_scale = (
            (
                cluster_weights[..., None].float()
                * slot_delta.square()
            ).sum(dim=1)
            / mass[..., None].clamp_min(1.0)
            + 1.0e-8
        ).sqrt()
        cluster_scale = cluster_scale.clamp_min(
            envelope_scale[:, None] * 0.1
        ).clamp_max(envelope_scale[:, None] * 0.5)
        slot_index = cluster_slot.clamp_min(0)[..., None].expand(-1, -1, 2)
        selected_center = cluster_centers.gather(
            1,
            slot_index,
        )
        selected_scale = cluster_scale.gather(
            1,
            slot_index,
        )
        assigned = cluster_slot.ge(0)[..., None]
        frame_center = torch.where(assigned, selected_center, frame_center)
        frame_scale = torch.where(assigned, selected_scale, frame_scale)
    return HierarchyCoordinatePlan(
        center=frame_center,
        scale=frame_scale.clamp_min(1.0e-4),
        envelope_center=envelope_center,
        envelope_scale=envelope_scale,
        cluster_slot=cluster_slot,
    )


def state_to_hierarchy_coordinates(
    block_state: torch.Tensor,
    plan: HierarchyCoordinatePlan,
) -> torch.Tensor:
    """Map global block centers into the selected hierarchy frame."""

    center = (block_state[..., :2].float() - plan.center) / plan.scale
    return torch.cat([center, block_state[..., 2:]], dim=-1)


def state_from_hierarchy_coordinates(
    local_state: torch.Tensor,
    plan: HierarchyCoordinatePlan,
) -> torch.Tensor:
    """Restore global block centers from a hierarchy-frame prediction."""

    center = local_state[..., :2].float() * plan.scale + plan.center
    return torch.cat([center, local_state[..., 2:]], dim=-1)


def _blend(blend: torch.Tensor | float, reference: torch.Tensor) -> torch.Tensor:
    value = torch.as_tensor(blend, dtype=reference.dtype, device=reference.device)
    return value.clamp(0.0, 1.0)


def parameterize_hierarchy_block_input(
    block_state: torch.Tensor,
    global_state: torch.Tensor,
    structural_state: torch.Tensor,
    membership: torch.Tensor | None,
    structural_mask: torch.Tensor | None,
    *,
    enabled: bool,
    cluster_local: bool = True,
    blend: torch.Tensor | float = 1.0,
) -> tuple[torch.Tensor, HierarchyCoordinatePlan | None]:
    """Apply opt-in envelope/cluster parameterization before the backbone."""

    if not enabled:
        return block_state, None
    plan = build_hierarchy_coordinate_plan(
        block_state,
        global_state,
        structural_state,
        membership,
        structural_mask,
        cluster_local=cluster_local,
    )
    local = state_to_hierarchy_coordinates(block_state, plan)
    strength = _blend(blend, block_state)
    return block_state + strength * (local - block_state), plan


def restore_hierarchy_coordinate_prediction(
    block_prediction: torch.Tensor,
    plan: HierarchyCoordinatePlan | None,
    blend: torch.Tensor | float = 1.0,
) -> torch.Tensor:
    """Restore a backbone prediction to the public global coordinate space."""

    if plan is None:
        return block_prediction
    restored = state_from_hierarchy_coordinates(block_prediction, plan)
    strength = _blend(blend, block_prediction)
    return block_prediction + strength * (restored - block_prediction)


__all__ = [
    "HierarchyBlockCoordinateGate",
    "HierarchyCoordinatePlan",
    "build_hierarchy_coordinate_plan",
    "parameterize_hierarchy_block_input",
    "restore_hierarchy_coordinate_prediction",
    "state_from_hierarchy_coordinates",
    "state_to_hierarchy_coordinates",
]
