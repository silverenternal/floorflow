"""Training-only endpoint projection and physical compatibility contracts."""

from __future__ import annotations

from argparse import Namespace

import torch
import torch.nn.functional as F

from floorflow.chip_flow.endpoint_compatibility import (
    endpoint_compatibility_terms,
)
from floorflow.chip_flow.hard_constraint_projector import (
    project_hard_constraints_latent,
)
from floorflow.chip_flow.objective_schedules import delayed_ramp_weight
from floorflow.chip_flow.projector_endpoint_contract import (
    projector_aware_endpoint,
)


def _apply_training_endpoint_contracts_fp32(
    raw_latent: torch.Tensor,
    batch: dict,
    args: Namespace,
    *,
    step: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Apply differentiable projection and physical endpoint losses."""

    latent = raw_latent
    total = torch.zeros((), device=device)
    terms: dict[str, torch.Tensor] = {}
    if bool(getattr(args, "training_projector", False)):
        projected = project_hard_constraints_latent(
            raw_latent,
            batch["target_latent"],
            batch["constraints"],
            batch["areas"],
            iterations=int(getattr(args, "training_projector_iterations", 2)),
            boundary_axes_locked=bool(
                getattr(args, "projector_boundary_axes", False)
            ),
            lock_boundary_corners=bool(
                getattr(args, "projector_boundary_lock_corners", True)
            ),
            sparse_pairs=bool(getattr(args, "projector_sparse_pairs", True)),
            projection_strategy=str(
                getattr(args, "projector_strategy", "legacy")
            ),
            grid_size=int(getattr(args, "projector_grid_size", 16)),
        )
        strength = float(getattr(args, "training_projector_strength", 1.0))
        latent = torch.lerp(raw_latent, projected, strength)
        displacement = F.smooth_l1_loss(
            projected - raw_latent,
            torch.zeros_like(raw_latent),
            beta=0.1,
        )
        displacement_weight = float(
            getattr(args, "training_projector_displacement_weight", 0.0)
        )
        total = total + displacement_weight * displacement
        terms.update(
            {
                "training_projector_displacement": displacement,
                "training_projector_active": torch.ones(
                    (),
                    device=device,
                ),
            }
        )

    endpoint_weight = delayed_ramp_weight(
        float(getattr(args, "endpoint_compatibility_weight", 0.0)),
        int(getattr(args, "endpoint_compatibility_delay_steps", 0)),
        int(getattr(args, "endpoint_compatibility_anneal_steps", 0)),
        step,
    )
    if endpoint_weight > 0.0:
        compatibility = endpoint_compatibility_terms(
            latent,
            batch,
            iterations=int(
                getattr(args, "endpoint_compatibility_iterations", 2)
            ),
            displacement_weight=float(
                getattr(args, "endpoint_compatibility_displacement_weight", 1.0)
            ),
            gain_weight=float(
                getattr(args, "endpoint_compatibility_gain_weight", 0.25)
            ),
            refiner_energy_weight=float(
                getattr(args, "endpoint_refiner_energy_weight", 1.0)
            ),
            refiner_steps=int(
                getattr(args, "endpoint_refiner_trajectory_steps", 2)
            ),
            refiner_lr=float(
                getattr(args, "endpoint_refiner_trajectory_lr", 1.0e-3)
            ),
            refiner_trajectory_weight=float(
                getattr(args, "endpoint_refiner_trajectory_weight", 1.0)
            ),
            legalized_area_weight=float(
                getattr(args, "endpoint_legalized_area_weight", 0.0)
            ),
            legalized_p2b_weight=float(
                getattr(args, "endpoint_legalized_p2b_weight", 0.0)
            ),
            legalizer_target_weight=float(
                getattr(args, "endpoint_legalizer_target_weight", 0.0)
            ),
            shape_mib_weight=float(
                getattr(args, "endpoint_mib_share_weight", 0.0)
            ),
            shape_pin_tail_weight=float(
                getattr(args, "endpoint_pin_tail_weight", 0.0)
            ),
            shape_rho_trust_weight=float(
                getattr(args, "endpoint_rho_trust_weight", 0.0)
            ),
            shape_rho_trust_radius=float(
                getattr(args, "endpoint_rho_trust_radius", 0.1)
            ),
            shape_pin_tail_fraction=float(
                getattr(args, "endpoint_pin_tail_fraction", 0.2)
            ),
        )
        total = total + endpoint_weight * compatibility.total
        terms.update(compatibility.as_dict())
        terms["endpoint_compatibility_current_weight"] = torch.tensor(
            endpoint_weight,
            device=device,
        )

    if bool(getattr(args, "projector_aware_endpoint", False)):
        latent, endpoint_terms = projector_aware_endpoint(
            latent,
            batch["target_latent"],
            batch["constraints"],
            batch["areas"],
            iterations=int(args.projector_aware_endpoint_iterations),
            straight_through=True,
            strength=float(args.projector_aware_endpoint_strength),
        )
        terms.update(endpoint_terms)
    return latent, total, terms


def apply_training_endpoint_contracts(
    raw_latent: torch.Tensor,
    batch: dict,
    args: Namespace,
    *,
    step: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Force second-order physical objectives outside BF16 autocast."""

    with torch.autocast(device_type=device.type, enabled=False):
        return _apply_training_endpoint_contracts_fp32(
            raw_latent.float(),
            batch,
            args,
            step=step,
            device=device,
        )


__all__ = ["apply_training_endpoint_contracts"]
