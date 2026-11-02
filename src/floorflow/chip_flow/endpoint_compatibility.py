"""Training-only compatibility targets for the existing projector and refiner."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from trinity.sampling.refine_closed import PAPER_WEIGHTS
from trinity.sampling.refine_constraint import (
    RefineCase,
    constraint_energy,
)

from floorflow.chip_flow.batched import batch_physics_terms
from floorflow.chip_flow.hard_constraint_projector import (
    project_hard_constraints_latent,
)
from floorflow.chip_flow.model import decode_boxes_batch
from floorflow.chip_flow.shape_pin_consistency import (
    shape_pin_consistency_terms,
)


def normalized_latent_displacement(
    latent: torch.Tensor,
    projected_latent: torch.Tensor,
) -> torch.Tensor:
    """Return center displacement in the already-normalized flow chart."""

    if latent.shape != projected_latent.shape:
        raise ValueError("latent and projected latent must share a shape")
    if latent.ndim != 3 or latent.shape[-1] != 3:
        raise ValueError("latents must have shape [B,N,3]")
    return projected_latent.float()[..., :2] - latent.float()[..., :2]


@dataclass(frozen=True)
class EndpointCompatibilityTerms:
    total: torch.Tensor
    raw_energy: torch.Tensor
    projected_energy: torch.Tensor
    displacement: torch.Tensor
    projection_gain: torch.Tensor
    refiner_energy: torch.Tensor
    refiner_trajectory_energy: torch.Tensor
    refiner_trajectory_displacement: torch.Tensor
    legalized_area: torch.Tensor
    legalized_p2b: torch.Tensor
    legalizer_target: torch.Tensor
    shape_pin: torch.Tensor
    mib_shared: torch.Tensor
    pin_tail: torch.Tensor
    rho_trust: torch.Tensor

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {
            "endpoint_compatibility": self.total,
            "endpoint_raw_energy": self.raw_energy,
            "endpoint_projected_energy": self.projected_energy,
            "endpoint_projection_displacement": self.displacement,
            "endpoint_projection_gain": self.projection_gain,
            "endpoint_refiner_energy": self.refiner_energy,
            "endpoint_refiner_trajectory_energy": self.refiner_trajectory_energy,
            "endpoint_refiner_trajectory_displacement": (
                self.refiner_trajectory_displacement
            ),
            "endpoint_legalized_area": self.legalized_area,
            "endpoint_legalized_p2b": self.legalized_p2b,
            "endpoint_legalizer_target": self.legalizer_target,
            "endpoint_shape_pin": self.shape_pin,
            "endpoint_mib_shared": self.mib_shared,
            "endpoint_pin_tail": self.pin_tail,
            "endpoint_rho_trust": self.rho_trust,
        }


def _differentiable_refiner_steps(
    latent: torch.Tensor,
    case: RefineCase,
    *,
    steps: int,
    lr: float,
    beta2: float = 0.99,
    eps: float = 1.0e-8,
    shape: bool = True,
) -> torch.Tensor:
    """Unroll the pinned closed-form energy for a few differentiable steps."""

    if steps < 0:
        raise ValueError("refiner trajectory steps must be non-negative")
    if steps == 0:
        return latent
    if lr < 0.0 or not 0.0 <= beta2 < 1.0:
        raise ValueError("invalid refiner trajectory optimizer settings")
    z = latent
    first_moment = torch.zeros_like(z)
    second_moment = torch.zeros_like(z)
    for step in range(1, steps + 1):
        source = z if z.requires_grad else z.detach().requires_grad_(True)
        energy = constraint_energy(source, case, PAPER_WEIGHTS).sum()
        gradient = torch.autograd.grad(energy, source, create_graph=True)[0]
        if shape:
            shape_mask = case.mob_shape
            gradient = torch.stack(
                [
                    gradient[..., 0] * case.mob_pos,
                    gradient[..., 1] * case.mob_pos,
                    gradient[..., 2] * shape_mask,
                ],
                dim=-1,
            )
        else:
            gradient = torch.stack(
                [
                    gradient[..., 0] * case.mob_pos,
                    gradient[..., 1] * case.mob_pos,
                    torch.zeros_like(gradient[..., 2]),
                ],
                dim=-1,
            )
        if case.anchor_mask is not None:
            gradient = gradient * (case.anchor_mask < 0.5)
        first_moment = beta2 * first_moment + (1.0 - beta2) * gradient
        second_moment = (
            beta2 * second_moment + (1.0 - beta2) * gradient.square()
        )
        first_hat = first_moment / (1.0 - beta2**step)
        second_hat = second_moment / (1.0 - beta2**step)
        denominator = (second_hat.clamp_min(0.0) + eps).sqrt()
        z = z - lr * first_hat / denominator
        if case.anchor_mask is not None:
            z = torch.where(case.anchor_mask > 0.5, case.anchor_z, z)
    return z


def _refine_case(batch: dict) -> RefineCase:
    areas = batch["areas"].float()
    constraints = batch["constraints"].float()
    scale = areas.sum(dim=1).sqrt().clamp_min(1.0e-6)
    batch_size, count = areas.shape
    b2b = batch["b2b"].float()
    n_b2b = batch["n_b2b"].long()
    rows = torch.arange(b2b.shape[1], device=b2b.device)[None]
    valid_b2b = rows < n_b2b[:, None]
    adjacency = b2b.new_zeros(batch_size, count, count)
    offsets = torch.arange(batch_size, device=b2b.device)[:, None] * count
    source = b2b[..., 0].long().clamp(0, count - 1) + offsets
    target = b2b[..., 1].long().clamp(0, count - 1)
    weights = b2b[..., 2].clamp_min(0.0)
    adjacency.reshape(-1).index_add_(
        0,
        (source * count + target).reshape(-1),
        (weights * valid_b2b).reshape(-1),
    )
    adjacency = adjacency + adjacency.transpose(1, 2)
    p2b = batch["p2b"].float()
    p2b_valid = (
        torch.arange(p2b.shape[1], device=p2b.device)[None]
        < batch["n_p2b"].long()[:, None]
    )
    pin_ids = p2b[..., 0].long().clamp(0, batch["pins"].shape[1] - 1)
    block_ids = p2b[..., 1].long().clamp(0, count - 1)
    pin_xy = batch["pins"].float().gather(
        1,
        pin_ids[..., None].expand(-1, -1, 2),
    )
    preplaced = constraints[..., 1].ne(0)
    fixed = constraints[..., 0].ne(0)
    anchor_z = torch.stack(
        [
            torch.where(preplaced, batch["target_latent"][..., 0], 0.0),
            torch.where(preplaced, batch["target_latent"][..., 1], 0.0),
            torch.where(
                fixed | preplaced,
                batch["target_latent"][..., 2],
                0.0,
            ),
        ],
        dim=-1,
    )
    anchor_mask = torch.stack([preplaced, preplaced, fixed | preplaced], dim=-1)
    return RefineCase(
        area_norm=areas / scale.square()[:, None],
        token_mask=torch.ones_like(areas),
        mob_pos=(~preplaced).float(),
        mob_shape=(~(fixed | preplaced)).float(),
        cluster_id=constraints[..., 3].long(),
        mib_id=constraints[..., 2].long(),
        boundary_code=constraints[..., 4].long(),
        adjacency=adjacency,
        pin_edge_xy_n=pin_xy / scale[:, None, None],
        pin_edge_w=p2b[..., 2] * p2b_valid,
        pin_edge_block=block_ids,
        anchor_z=anchor_z,
        anchor_mask=anchor_mask.float(),
        outline=None,
    )


def endpoint_compatibility_terms(
    latent: torch.Tensor,
    batch: dict,
    *,
    iterations: int = 2,
    displacement_weight: float = 1.0,
    gain_weight: float = 0.25,
    refiner_energy_weight: float = 1.0,
    refiner_steps: int = 2,
    refiner_lr: float = 1.0e-3,
    refiner_trajectory_weight: float = 1.0,
    legalized_area_weight: float = 0.0,
    legalized_p2b_weight: float = 0.0,
    legalizer_target_weight: float = 0.0,
    shape_mib_weight: float = 0.0,
    shape_pin_tail_weight: float = 0.0,
    shape_rho_trust_weight: float = 0.0,
    shape_rho_trust_radius: float = 0.1,
    shape_pin_tail_fraction: float = 0.2,
) -> EndpointCompatibilityTerms:
    """Measure whether an endpoint is already close to the legalized solution.

    The objective is used during training only. Inference keeps the established
    refiner, hard projector and scale_pack legalizer unchanged.
    """

    if iterations < 0:
        raise ValueError("endpoint compatibility iterations must be non-negative")
    if (
        displacement_weight < 0.0
        or gain_weight < 0.0
        or refiner_energy_weight < 0.0
        or refiner_trajectory_weight < 0.0
        or legalized_area_weight < 0.0
        or legalized_p2b_weight < 0.0
        or legalizer_target_weight < 0.0
        or shape_mib_weight < 0.0
        or shape_pin_tail_weight < 0.0
        or shape_rho_trust_weight < 0.0
    ):
        raise ValueError("endpoint compatibility weights must be non-negative")
    raw_boxes = decode_boxes_batch(latent.float(), batch["areas"].float())
    if iterations == 0:
        projected_latent = latent.float()
    else:
        with torch.no_grad():
            projected_latent = project_hard_constraints_latent(
                latent.detach().float(),
                batch["target_latent"].float(),
                batch["constraints"],
                batch["areas"].float(),
                iterations=iterations,
            )
        projected_latent = latent.float() + (
            projected_latent - latent.float()
        ).detach()
    projected_boxes = decode_boxes_batch(
        projected_latent,
        batch["areas"].float(),
    )

    def physics(boxes: torch.Tensor) -> dict[str, torch.Tensor]:
        terms = batch_physics_terms(
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
        return {name: value.mean() for name, value in terms.items()}

    raw_physics = physics(raw_boxes)
    projected_physics = physics(projected_boxes)
    raw_energy = raw_physics["total"]
    projected_energy = projected_physics["total"]
    refine_case = _refine_case(batch)
    refiner_energy = constraint_energy(
        latent.float(),
        refine_case,
        PAPER_WEIGHTS,
    ).mean()
    refined = _differentiable_refiner_steps(
        latent.float(),
        refine_case,
        steps=refiner_steps,
        lr=refiner_lr,
    )
    refiner_trajectory_energy = constraint_energy(
        refined,
        refine_case,
        PAPER_WEIGHTS,
    ).mean()
    refiner_trajectory_displacement = F.smooth_l1_loss(
        refined - latent.float(),
        torch.zeros_like(latent),
        beta=0.1,
    )
    displacement = F.smooth_l1_loss(
        normalized_latent_displacement(latent, projected_latent),
        torch.zeros_like(latent[..., :2]),
        beta=0.1,
    )
    projection_gain = (raw_energy - projected_energy).clamp_min(0.0)
    target = batch.get("legalizer_displacement")
    target_mask = batch.get("legalizer_mask")
    if (
        legalizer_target_weight > 0.0
        and target is not None
        and target_mask is not None
    ):
        target_projected_latent = project_hard_constraints_latent(
            latent.float(),
            batch["target_latent"].float(),
            batch["constraints"],
            batch["areas"].float(),
            iterations=iterations,
        )
        normalized = normalized_latent_displacement(
            latent,
            target_projected_latent,
        )
        target_error = F.smooth_l1_loss(
            normalized,
            target.float(),
            beta=0.1,
            reduction="none",
        ).mean(dim=-1)
        legalizer_target = (
            target_error * target_mask.float()
        ).sum() / target_mask.float().sum().clamp_min(1.0)
    else:
        legalizer_target = torch.zeros((), device=latent.device)
    shape_pin = shape_pin_consistency_terms(
        projected_boxes,
        raw_boxes,
        batch["areas"].float(),
        batch["constraints"],
        batch["p2b"],
        batch["pins"],
        batch["n_p2b"],
        mib_weight=shape_mib_weight,
        pin_tail_weight=shape_pin_tail_weight,
        rho_trust_weight=shape_rho_trust_weight,
        rho_trust_radius=shape_rho_trust_radius,
        pin_tail_fraction=shape_pin_tail_fraction,
    )
    total = (
        projected_energy
        + float(displacement_weight) * displacement
        + float(gain_weight) * projection_gain
        + float(refiner_energy_weight) * refiner_energy
        + float(refiner_trajectory_weight)
        * (refiner_trajectory_energy + refiner_trajectory_displacement)
        + float(legalized_area_weight) * projected_physics["area"]
        + float(legalized_p2b_weight) * projected_physics["p2b"]
        + float(legalizer_target_weight) * legalizer_target
        + shape_pin.total
    )
    return EndpointCompatibilityTerms(
        total=total,
        raw_energy=raw_energy,
        projected_energy=projected_energy,
        displacement=displacement,
        projection_gain=projection_gain,
        refiner_energy=refiner_energy,
        refiner_trajectory_energy=refiner_trajectory_energy,
        refiner_trajectory_displacement=refiner_trajectory_displacement,
        legalized_area=projected_physics["area"],
        legalized_p2b=projected_physics["p2b"],
        legalizer_target=legalizer_target,
        shape_pin=shape_pin.total,
        mib_shared=shape_pin.mib_shared,
        pin_tail=shape_pin.pin_tail,
        rho_trust=shape_pin.rho_trust,
    )


__all__ = [
    "EndpointCompatibilityTerms",
    "endpoint_compatibility_terms",
    "normalized_latent_displacement",
]
