"""Current-block observations for coarse-to-fine hierarchy coupling."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from floorflow.chip_flow.model import decode_boxes_batch
from floorflow.chip_flow.sparse_layers import masked_token_mean


@dataclass(frozen=True)
class ObservedHierarchyState:
    """L0/L1/L3 states observed from the current L2 block layout."""

    global_state: torch.Tensor
    structural_state: torch.Tensor
    port_center: torch.Tensor


class ObservedHierarchyCoupler(nn.Module):
    """Zero-initialized L2 observations injected into L0/L1/L3 tokens."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        if hidden <= 0:
            raise ValueError("hidden must be positive")
        self.global_adapter = nn.Linear(4, hidden)
        self.structural_adapter = nn.Linear(3, hidden)
        self.port_adapter = nn.Linear(2, hidden)
        for adapter in (
            self.global_adapter,
            self.structural_adapter,
            self.port_adapter,
        ):
            nn.init.zeros_(adapter.weight)
            nn.init.zeros_(adapter.bias)

    def forward(
        self,
        block_state: torch.Tensor,
        global_token: torch.Tensor,
        structural_tokens: torch.Tensor,
        port_tokens: torch.Tensor,
        areas: torch.Tensor,
        membership: torch.Tensor,
        structural_mask: torch.Tensor,
        port_incidence: torch.Tensor,
        port_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, ObservedHierarchyState]:
        observed = observed_hierarchy_state(
            block_state,
            areas,
            membership,
            structural_mask,
            port_incidence,
            port_mask,
        )
        return (
            global_token
            + self.global_adapter(observed.global_state)[:, None],
            structural_tokens
            + self.structural_adapter(observed.structural_state),
            port_tokens + self.port_adapter(observed.port_center),
            observed,
        )


def observed_hierarchy_state(
    block_state: torch.Tensor,
    areas: torch.Tensor,
    membership: torch.Tensor,
    structural_mask: torch.Tensor,
    port_incidence: torch.Tensor,
    port_mask: torch.Tensor,
) -> ObservedHierarchyState:
    """Aggregate a block layout into coarse and port observations."""

    if block_state.ndim != 3 or block_state.shape[-1] != 3:
        raise ValueError("block_state must be [B,N,3]")
    batch, count, _ = block_state.shape
    if areas.shape != (batch, count):
        raise ValueError("areas must match block_state")
    if membership.ndim != 3 or membership.shape[:2] != (batch, count):
        raise ValueError("membership must match block_state")
    if structural_mask.shape != (batch, membership.shape[-1]):
        raise ValueError("structural_mask must match membership slots")
    if port_incidence.shape != (
        batch,
        count,
        port_mask.shape[1],
    ) or port_mask.shape[0] != batch:
        raise ValueError("port incidence and mask shapes do not match")

    boxes = decode_boxes_batch(block_state.float(), areas.float())
    scale = areas.float().sum(dim=1).sqrt().clamp_min(1.0e-6)
    lower = boxes[..., :2].amin(dim=1)
    upper = (boxes[..., :2] + boxes[..., 2:]).amax(dim=1)
    global_state = torch.stack(
        [
            (lower[:, 0] + upper[:, 0]) * 0.5 / scale,
            (lower[:, 1] + upper[:, 1]) * 0.5 / scale,
            (upper[:, 0] - lower[:, 0]) / scale,
            (upper[:, 1] - lower[:, 1]) / scale,
        ],
        dim=-1,
    )
    structural_state = torch.stack(
        [
            masked_token_mean(block_state.float(), membership[..., index])
            for index in range(membership.shape[-1])
        ],
        dim=1,
    )
    structural_state = structural_state * structural_mask[..., None]
    center = block_state.float()[..., :2]
    weights = (
        port_incidence.float()
        * port_mask[:, None, :].float()
    )
    port_state = torch.bmm(
        weights.transpose(1, 2),
        center,
    ) / weights.sum(dim=1)[..., None].clamp_min(1.0)
    return ObservedHierarchyState(
        global_state=global_state,
        structural_state=structural_state,
        port_center=port_state,
    )


__all__ = [
    "ObservedHierarchyCoupler",
    "ObservedHierarchyState",
    "observed_hierarchy_state",
]
