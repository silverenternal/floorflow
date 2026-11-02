"""Differentiable approximation of the official exponential FloorSet cost."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from floorflow.chip_flow.batched import (
    batch_physics_terms,
    differentiable_pair_gap,
)
from floorflow.chip_flow.model import decode_boxes_batch
from floorflow.chip_flow.objective_schedules import delayed_ramp_weight

TAU_GROUP = 0.01
TAU_MIB = 0.05
TAU_BOUNDARY = 0.01
CLUSTER_GROUPS = 4
MIB_GROUPS = 1


@dataclass(frozen=True)
class OfficialCostTerms:
    total: torch.Tensor
    hpwl_gap: torch.Tensor
    area_gap: torch.Tensor
    soft_v_rel: torch.Tensor
    grouping: torch.Tensor
    boundary: torch.Tensor
    mib: torch.Tensor
    n_soft: torch.Tensor

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {
            "official_cost_surrogate": self.total,
            "official_cost_hpwl_gap": self.hpwl_gap,
            "official_cost_area_gap": self.area_gap,
            "official_cost_soft_v_rel": self.soft_v_rel,
            "official_cost_grouping": self.grouping,
            "official_cost_boundary": self.boundary,
            "official_cost_mib": self.mib,
            "official_cost_n_soft": self.n_soft,
        }


def _physics(batch: dict, boxes: torch.Tensor) -> dict[str, torch.Tensor]:
    return batch_physics_terms(
        boxes,
        batch["areas"].float(),
        batch["constraints"],
        batch["b2b"],
        batch["p2b"],
        batch["pins"],
        batch["n_b2b"],
        batch["n_p2b"],
        batch["n_pins"],
    )


def _drop_smallest_slot_sum(
    values: torch.Tensor,
    membership: torch.Tensor,
    group_count: int,
) -> torch.Tensor:
    """Sum one severity slot per member while dropping the easiest member."""

    total = values.new_zeros(values.shape[0])
    fill = torch.finfo(values.dtype).max
    for group in range(1, group_count + 1):
        selected = membership.eq(group)
        count = selected.sum(dim=1)
        selected_sum = values.masked_fill(~selected, 0.0).sum(dim=1)
        smallest = values.masked_fill(
            ~selected,
            fill,
        ).amin(dim=1)
        total = total + torch.where(
            count.gt(1),
            selected_sum - smallest,
            torch.zeros_like(total),
        )
    return total


def differentiable_soft_severity(
    boxes: torch.Tensor,
    areas: torch.Tensor,
    constraints: torch.Tensor,
    *,
    tau_group: float = TAU_GROUP,
    tau_mib: float = TAU_MIB,
    tau_boundary: float = TAU_BOUNDARY,
) -> dict[str, torch.Tensor]:
    """Approximate Trinity's severity-graded V_rel with differentiable slots."""

    if boxes.ndim != 3 or boxes.shape[-1] != 4:
        raise ValueError("boxes must have shape [B,N,4]")
    if areas.shape != boxes.shape[:2]:
        raise ValueError("areas must match the box block axes")
    if constraints.shape != (*areas.shape, 5):
        raise ValueError("constraints must have shape [B,N,5]")
    if min(tau_group, tau_mib, tau_boundary) <= 0.0:
        raise ValueError("soft-severity temperatures must be positive")

    batch, count = areas.shape
    scale = areas.float().sum(dim=1).sqrt().clamp_min(1.0e-6)
    pair_gap = (
        differentiable_pair_gap(boxes.float()) / scale[:, None, None]
    )
    cluster = constraints[..., 3].long()
    same_cluster = cluster[:, :, None].eq(cluster[:, None, :])
    same_cluster &= cluster[:, :, None].gt(0)
    same_cluster &= ~torch.eye(
        count,
        dtype=torch.bool,
        device=boxes.device,
    )[None]
    nearest_gap = pair_gap.masked_fill(
        ~same_cluster,
        torch.finfo(pair_gap.dtype).max,
    ).amin(dim=-1)
    nearest_gap = torch.where(
        same_cluster.any(dim=-1),
        nearest_gap,
        torch.zeros_like(nearest_gap),
    )
    grouping = _drop_smallest_slot_sum(
        torch.tanh(nearest_gap / float(tau_group)),
        cluster,
        CLUSTER_GROUPS,
    )

    log_width = boxes[..., 2].float().clamp_min(1.0e-6).log()
    log_height = boxes[..., 3].float().clamp_min(1.0e-6).log()
    mib = constraints[..., 2].long()
    mib_slots = boxes.new_zeros(batch, count)
    for group in range(1, MIB_GROUPS + 1):
        selected = mib.eq(group)
        count_group = selected.sum(dim=1).clamp_min(1)
        mean_w = (
            log_width.masked_fill(~selected, 0.0).sum(dim=1)
            / count_group
        )
        mean_h = (
            log_height.masked_fill(~selected, 0.0).sum(dim=1)
            / count_group
        )
        deviation = (
            (log_width - mean_w[:, None]).abs()
            + (log_height - mean_h[:, None]).abs()
        ).masked_fill(~selected, 0.0)
        mib_slots = mib_slots + deviation
    mib_severity = _drop_smallest_slot_sum(
        torch.tanh(mib_slots / float(tau_mib)),
        mib,
        MIB_GROUPS,
    )

    x0 = boxes[..., 0].float()
    y0 = boxes[..., 1].float()
    x1 = x0 + boxes[..., 2].float()
    y1 = y0 + boxes[..., 3].float()
    x_min = x0.amin(dim=1)
    y_min = y0.amin(dim=1)
    x_max = x1.amax(dim=1)
    y_max = y1.amax(dim=1)
    codes = constraints[..., 4].long()
    distance = (
        (codes.bitwise_and(1).ne(0) * (x0 - x_min[:, None]).abs())
        + (codes.bitwise_and(2).ne(0) * (x1 - x_max[:, None]).abs())
        + (codes.bitwise_and(4).ne(0) * (y1 - y_max[:, None]).abs())
        + (codes.bitwise_and(8).ne(0) * (y0 - y_min[:, None]).abs())
    )
    boundary = torch.where(
        codes.ne(0),
        torch.tanh(distance / scale[:, None] / float(tau_boundary)),
        torch.zeros_like(distance),
    ).sum(dim=1)

    n_soft = codes.ne(0).to(areas.dtype).sum(dim=1)
    for group in range(1, CLUSTER_GROUPS + 1):
        n_soft = n_soft + (cluster.eq(group).sum(dim=1) - 1).clamp_min(0)
    for group in range(1, MIB_GROUPS + 1):
        n_soft = n_soft + (mib.eq(group).sum(dim=1) - 1).clamp_min(0)
    severity_relative = (
        grouping + mib_severity + boundary
    ) / n_soft.clamp_min(1.0)
    return {
        "grouping": grouping.float(),
        "mib": mib_severity.float(),
        "boundary": boundary.float(),
        "n_soft": n_soft.float(),
        "severity_relative": severity_relative.float(),
    }


def official_cost_surrogate(
    boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    batch: dict,
    *,
    violation_temperature: float = 0.1,
) -> OfficialCostTerms:
    """Build a smooth, target-conditioned approximation of official hard cost."""

    if boxes.shape != target_boxes.shape:
        raise ValueError("predicted and target boxes must have matching shapes")
    if violation_temperature <= 0.0:
        raise ValueError("violation temperature must be positive")
    predicted = _physics(batch, boxes.float())
    target = _physics(batch, target_boxes.float())
    eps = 1.0e-6
    hpwl_gap = (
        predicted["wl"] - target["wl"]
    ).clamp_min(0.0) / target["wl"].detach().clamp_min(eps)
    area_gap = (
        predicted["area"] - target["area"]
    ).clamp_min(0.0) / target["area"].detach().clamp_min(eps)
    temperature_scale = float(violation_temperature) / 0.1
    severity = differentiable_soft_severity(
        boxes.float(),
        batch["areas"],
        batch["constraints"],
        tau_group=TAU_GROUP * temperature_scale,
        tau_mib=TAU_MIB * temperature_scale,
        tau_boundary=TAU_BOUNDARY * temperature_scale,
    )
    soft_v_rel = severity["severity_relative"].clamp(0.0, 1.0)
    hpwl_gap = hpwl_gap.clamp(0.0, 10.0)
    area_gap = area_gap.clamp(0.0, 10.0)
    total = (
        1.0 + 0.5 * (hpwl_gap + area_gap)
    ) * torch.exp(2.0 * soft_v_rel)
    return OfficialCostTerms(
        total=total.mean(),
        hpwl_gap=hpwl_gap.mean(),
        area_gap=area_gap.mean(),
        soft_v_rel=soft_v_rel.mean(),
        grouping=severity["grouping"].mean(),
        boundary=severity["boundary"].mean(),
        mib=severity["mib"].mean(),
        n_soft=severity["n_soft"].mean(),
    )


def official_cost_objective(
    boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    batch: dict,
    args,
    *,
    step: int = 0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    weight = float(getattr(args, "official_cost_weight", 0.0))
    final_weight = getattr(args, "official_cost_final_weight", None)
    if final_weight is not None:
        weight = delayed_ramp_weight(
            float(final_weight),
            int(getattr(args, "official_cost_delay_steps", 0)),
            int(getattr(args, "official_cost_anneal_steps", 0)),
            step,
        )
    if weight <= 0.0:
        return boxes.new_zeros(()), {}
    terms = official_cost_surrogate(
        boxes,
        target_boxes,
        batch,
        violation_temperature=float(
            getattr(args, "official_cost_violation_temperature", 0.1)
        ),
    )
    return weight * terms.total, {
        **terms.as_dict(),
        "official_cost_current_weight": boxes.new_tensor(weight),
    }


def endpoint_boxes_from_latent(
    latent: torch.Tensor,
    areas: torch.Tensor,
) -> torch.Tensor:
    """Compatibility wrapper around the shared exact-area decoder."""

    return decode_boxes_batch(latent, areas)


__all__ = [
    "OfficialCostTerms",
    "differentiable_soft_severity",
    "endpoint_boxes_from_latent",
    "official_cost_objective",
    "official_cost_surrogate",
]
