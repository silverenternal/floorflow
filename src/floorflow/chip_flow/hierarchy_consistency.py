"""Self-consistency between decoded chip geometry and hierarchy heads."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from floorflow.chip_flow.hierarchical_flow_runtime import block_global_state
from floorflow.chip_flow.hierarchy_state import HierarchyPrediction
from floorflow.chip_flow.hierarchy_structure import structural_membership
from floorflow.chip_flow.sparse_layers import masked_token_mean


def _masked_smooth_l1(
    prediction: torch.Tensor,
    observed: torch.Tensor,
    mask: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    expanded = mask[..., None].expand_as(prediction)
    loss = F.smooth_l1_loss(
        prediction.float(),
        observed.float(),
        beta=beta,
        reduction="none",
    )
    denominator = expanded.float().sum().clamp_min(1.0)
    return (loss * expanded).sum() / denominator


def hierarchy_cycle_loss(
    prediction: HierarchyPrediction,
    boxes: torch.Tensor,
    batch: dict,
    context,
    *,
    huber_beta: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Recompute L0/L1/L3 from the endpoint and match predicted states."""

    latent = prediction.block_state.float()
    observed_global = block_global_state(
        boxes.float(),
        batch["areas"].float(),
    )
    global_loss = F.smooth_l1_loss(
        prediction.global_state.float(),
        observed_global,
        beta=huber_beta,
    )
    membership = structural_membership(batch["constraints"])
    observed_structural = torch.stack(
        [
            masked_token_mean(latent, membership[..., index])
            for index in range(membership.shape[-1])
        ],
        dim=1,
    )
    structural_loss = _masked_smooth_l1(
        prediction.structural_state,
        observed_structural,
        context.typed.structural_mask,
        huber_beta,
    )
    incidence = context.typed.port_block_incidence.float()
    degree = incidence.sum(dim=1).clamp_min(1.0)
    weighted = torch.bmm(
        incidence.transpose(1, 2),
        latent[..., :2],
    )
    block_center = weighted / degree[..., None]
    scale = batch["areas"].float().sum(dim=1).sqrt().clamp_min(1.0e-6)
    observed_port = (
        block_center - batch["pins"].float() / scale[:, None, None]
    )
    port_loss = _masked_smooth_l1(
        prediction.port_state,
        observed_port,
        context.typed.port_mask,
        huber_beta,
    )
    total = global_loss + structural_loss + port_loss
    return total, {
        "hierarchy_cycle": total,
        "hierarchy_cycle_global": global_loss,
        "hierarchy_cycle_structural": structural_loss,
        "hierarchy_cycle_port": port_loss,
    }


__all__ = ["hierarchy_cycle_loss"]
