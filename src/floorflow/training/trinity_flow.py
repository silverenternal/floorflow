"""Adapters for Trinity's framing, time sampler and room-latent objectives."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

import trinity  # noqa: F401  (populate registries)
from trinity.registry import FRAMING, TIME_SAMPLER, build

from floorflow.objectives.trinity_constraints import (
    TrinityConstraintWeights,
    trinity_constraint_loss,
)
from floorflow.representation.trinity_latent import TrinityRoomBatch


@dataclass(frozen=True)
class RoomFlowBatch:
    x_t: torch.Tensor
    target: torch.Tensor
    noise: torch.Tensor
    time: torch.Tensor


def make_trinity_flow(
    framing: str | dict = "rectified_flow",
    time_sampler: str | dict = {"name": "logit_normal"},
):
    return build(framing, FRAMING), build(time_sampler, TIME_SAMPLER)


def prepare_room_flow(
    z0: torch.Tensor,
    token_mask: torch.Tensor,
    *,
    framing=None,
    time_sampler=None,
) -> RoomFlowBatch:
    if framing is None or time_sampler is None:
        framing, time_sampler = make_trinity_flow()
    mask = token_mask.to(z0.dtype)
    noise = torch.randn_like(z0) * mask.unsqueeze(-1)
    time = time_sampler(z0.shape[0], z0.device)
    x_t, target = framing.prepare(z0, noise, time)
    return RoomFlowBatch(x_t=x_t, target=target, noise=noise, time=time)


def clamp_room_latent(
    z: torch.Tensor,
    token_mask: torch.Tensor,
    *,
    center_min: float = -2.0,
    center_max: float = 3.0,
    rho_limit: float = 4.0,
) -> torch.Tensor:
    """Keep recovered room latents in the finite physical domain."""

    z = torch.nan_to_num(z, nan=0.0, posinf=center_max, neginf=center_min)
    center = z[..., :2].clamp(center_min, center_max)
    rho = z[..., 2:3].clamp(-rho_limit, rho_limit)
    return torch.cat([center, rho], dim=-1) * token_mask.to(z.dtype).unsqueeze(-1)


def room_flow_objective(
    prediction: torch.Tensor,
    z0: torch.Tensor,
    batch: TrinityRoomBatch,
    *,
    framing=None,
    time_sampler=None,
    flow_weight: float = 1.0,
    constraint_weight: float = 1.0,
    constraint_weights: TrinityConstraintWeights | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Trinity framing flow loss plus room-domain differentiable physics."""

    if framing is None or time_sampler is None:
        framing, time_sampler = make_trinity_flow()
    flow = prepare_room_flow(
        z0,
        batch.token_mask,
        framing=framing,
        time_sampler=time_sampler,
    )
    mask = batch.token_mask.to(prediction.dtype).unsqueeze(-1)
    flow_error = F.smooth_l1_loss(
        prediction * mask,
        flow.target * mask,
        reduction="none",
    )
    flow_loss = flow_error.sum() / mask.expand_as(flow_error).sum().clamp_min(1.0)
    z0_hat = clamp_room_latent(
        framing.x0_from_target(flow.x_t, prediction, flow.time),
        batch.token_mask,
    )
    physics_loss, physics_components = trinity_constraint_loss(
        z0_hat,
        batch,
        weights=constraint_weights,
        time=flow.time,
    )
    total = float(flow_weight) * flow_loss + float(constraint_weight) * physics_loss
    components = {
        "flow": flow_loss.detach(),
        "physics": physics_loss.detach(),
        "total": total.detach(),
        **physics_components,
    }
    return total, components


def match_room_tokens(
    z0_prediction: torch.Tensor,
    class_logits: torch.Tensor,
    log_area: torch.Tensor,
    batch: TrinityRoomBatch,
) -> dict[str, torch.Tensor]:
    """Permutation-invariant room-token matching for supervised training."""

    batch_size, room_tokens = batch.token_mask.shape
    target_for_prediction = torch.full(
        (batch_size, room_tokens),
        -1,
        device=z0_prediction.device,
        dtype=torch.long,
    )
    matched_mask = torch.zeros_like(batch.token_mask, dtype=torch.bool)
    for index in range(batch_size):
        valid = torch.nonzero(batch.token_mask[index], as_tuple=False).flatten()
        if valid.numel() == 0:
            continue
        target_z = batch.z[index, valid]
        geometry_cost = torch.cdist(target_z, z0_prediction[index]).square()
        class_cost = -F.log_softmax(class_logits[index], dim=-1)[
            :,
            batch.room_classes[index, valid],
        ].transpose(0, 1)
        area_cost = (
            log_area[index][None, :]
            - torch.log1p(batch.area_targets[index, valid])[:, None]
        ).abs()
        cost = geometry_cost + class_cost + 0.1 * area_cost
        rows, columns = linear_sum_assignment(cost.detach().cpu().numpy())
        for row, column in zip(rows, columns):
            target_index = int(valid[int(row)])
            target_for_prediction[index, int(column)] = target_index
            matched_mask[index, int(column)] = True
    return {
        "target_for_prediction": target_for_prediction,
        "matched_mask": matched_mask,
    }


def gather_matched_room_targets(
    values: torch.Tensor,
    target_for_prediction: torch.Tensor,
) -> torch.Tensor:
    """Scatter target-order values into the matched prediction-token order."""

    output = torch.zeros_like(values)
    for index in range(values.shape[0]):
        targets = target_for_prediction[index]
        valid = targets >= 0
        output[index, valid] = values[index, targets[valid]]
    return output


__all__ = [
    "RoomFlowBatch",
    "clamp_room_latent",
    "gather_matched_room_targets",
    "match_room_tokens",
    "make_trinity_flow",
    "prepare_room_flow",
    "room_flow_objective",
]
