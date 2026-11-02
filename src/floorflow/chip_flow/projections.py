"""Input-conditioned latent projections shared by all chip sampling arms."""

from __future__ import annotations

from dataclasses import dataclass

import torch


class _MaskedLowerMedian(torch.autograd.Function):
    """Lower median over masked rows with ``torch.median`` gradient semantics."""

    @staticmethod
    def forward(ctx, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        masked = torch.where(
            mask,
            values,
            torch.full_like(values, float("inf")),
        )
        sorted_values = masked.sort(dim=-1).values
        counts = mask.sum(dim=-1)
        index = ((counts - 1).clamp_min(0) // 2)[..., None]
        median = sorted_values.gather(-1, index).squeeze(-1)
        ctx.save_for_backward(values, mask, median)
        return median

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        values, mask, median = ctx.saved_tensors
        selected = mask & values.eq(median[..., None])
        selected_count = selected.sum(dim=-1, keepdim=True).clamp_min(1)
        grad_values = grad_output[..., None] * selected / selected_count
        return grad_values, None


def _masked_lower_median(
    values: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    return _MaskedLowerMedian.apply(values, mask)


@dataclass(frozen=True)
class ConditionedProjectionPlan:
    """Invariant masks and target values reused across projector iterations."""

    anchored: torch.Tensor
    preplaced: torch.Tensor
    target_rho: torch.Tensor
    preplaced_xy: torch.Tensor
    member: torch.Tensor
    membership: torch.Tensor | None
    target_median: torch.Tensor | None
    anchor_counts: torch.Tensor | None
    member_counts: torch.Tensor | None

    def apply(self, latent: torch.Tensor) -> torch.Tensor:
        rho = torch.where(
            self.anchored,
            self.target_rho,
            latent[..., 2],
        )
        xy = torch.where(
            self.preplaced[..., None],
            self.preplaced_xy,
            latent[..., :2],
        )
        if self.membership is None or self.target_median is None:
            return torch.cat([xy, rho[..., None]], dim=-1)
        latent_members = torch.where(
            self.membership,
            rho[..., None].expand_as(self.membership),
            torch.zeros_like(self.membership, dtype=latent.dtype),
        ).transpose(1, 2)
        latent_median = _masked_lower_median(
            latent_members,
            self.membership.transpose(1, 2),
        )
        group_rho = torch.where(
            self.anchor_counts > 0,
            self.target_median,
            latent_median,
        )
        group_rho = torch.where(
            self.member_counts > 0,
            group_rho,
            torch.zeros_like(group_rho),
        )
        block_rho = (self.membership.to(group_rho.dtype) * group_rho[:, None]).sum(
            -1
        )
        rho = torch.where(
            self.member,
            block_rho,
            rho,
        )
        return torch.cat([xy, rho[..., None]], dim=-1)


def prepare_conditioned_projection(
    target: torch.Tensor,
    constraints: torch.Tensor,
) -> ConditionedProjectionPlan:
    """Precompute the invariant part of fixed/preplaced/MIB projection."""

    fixed = constraints[..., 0] != 0
    preplaced = constraints[..., 1] != 0
    anchored = fixed | preplaced
    mib_ids = constraints[..., 2]
    member = mib_ids > 0
    membership = None
    target_median = None
    anchor_counts = None
    member_counts = None
    if member.any():
        groups = torch.unique(mib_ids[member], sorted=True)
        membership = (mib_ids[..., None] == groups) & member[..., None]
        anchor_membership = membership & anchored[..., None]
        target_members = torch.where(
            anchor_membership,
            target[..., 2, None].expand_as(membership),
            torch.zeros_like(membership, dtype=target.dtype),
        ).transpose(1, 2)
        target_median = _masked_lower_median(
            target_members,
            anchor_membership.transpose(1, 2),
        )
        anchor_counts = anchor_membership.sum(dim=1)
        member_counts = membership.sum(dim=1)
    return ConditionedProjectionPlan(
        anchored=anchored,
        preplaced=preplaced,
        target_rho=target[..., 2],
        preplaced_xy=target[..., :2],
        member=member,
        membership=membership,
        target_median=target_median,
        anchor_counts=anchor_counts,
        member_counts=member_counts,
    )


def project_conditioned_latent(
    latent: torch.Tensor,
    target: torch.Tensor,
    constraints: torch.Tensor,
) -> torch.Tensor:
    """Apply fixed, preplaced and anchored-MIB projections.

    ``target`` is used only at rows marked fixed or preplaced, whose shape or
    position is an input condition. No free-block test target is consulted.
    """

    return prepare_conditioned_projection(target, constraints).apply(latent)


__all__ = [
    "ConditionedProjectionPlan",
    "prepare_conditioned_projection",
    "project_conditioned_latent",
]
