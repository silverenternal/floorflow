"""Training-only targets for the coupled chip hierarchy state."""

from __future__ import annotations

import torch

from floorflow.chip_flow.hierarchy_state import HierarchyState
from floorflow.chip_flow.hierarchy_structure import structural_membership
from floorflow.chip_flow.sparse_layers import masked_token_mean
from floorflow.chip_flow.structural_relations import structural_relation_features


def build_hierarchy_targets(
    batch: dict,
    context,
) -> HierarchyState:
    """Derive training targets for all levels; context stays target-free."""

    boxes = batch["boxes"].float()
    areas = batch["areas"].float()
    latent = batch["target_latent"].float()
    scale = areas.sum(dim=1).sqrt().clamp_min(1.0e-6)
    x0 = boxes[..., 0].amin(dim=1)
    y0 = boxes[..., 1].amin(dim=1)
    x1 = (boxes[..., 0] + boxes[..., 2]).amax(dim=1)
    y1 = (boxes[..., 1] + boxes[..., 3]).amax(dim=1)
    global_target = torch.stack(
        [
            (x0 + x1) * 0.5 / scale,
            (y0 + y1) * 0.5 / scale,
            (x1 - x0) / scale,
            (y1 - y0) / scale,
        ],
        dim=-1,
    )
    membership = structural_membership(batch["constraints"])
    structural_target = torch.stack(
        [
            masked_token_mean(latent, membership[..., index])
            for index in range(membership.shape[-1])
        ],
        dim=1,
    )
    incidence = context.typed.port_block_incidence
    degree = incidence.float().sum(dim=1).clamp_min(1.0)
    weighted = torch.bmm(
        incidence.float().transpose(1, 2),
        latent[..., :2],
    )
    block_center = weighted / degree[..., None]
    port_target = block_center - batch["pins"].float() / scale[:, None, None]
    port_target = port_target * context.typed.port_mask[..., None]
    moment_target = None
    if context.native_group_moment_state:
        moment_target, _ = structural_relation_features(
            boxes,
            areas,
            batch["constraints"],
        )
    return HierarchyState(
        global_state=global_target,
        structural_state=structural_target,
        block_state=latent,
        port_state=port_target,
        structural_moment_state=moment_target,
    )


__all__ = ["build_hierarchy_targets"]
