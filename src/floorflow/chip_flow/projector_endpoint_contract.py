"""Training-time endpoint contract through the deployment hard projector."""

from __future__ import annotations

import torch

from floorflow.chip_flow.hard_constraint_projector import (
    project_hard_constraints_latent,
)


def projector_aware_endpoint(
    latent: torch.Tensor,
    target_latent: torch.Tensor,
    constraints: torch.Tensor,
    areas: torch.Tensor,
    *,
    iterations: int,
    straight_through: bool,
    strength: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Expose projected geometry while preserving gradients to the raw x0."""

    if not 0.0 <= strength <= 1.0:
        raise ValueError("projector endpoint strength must be in [0, 1]")
    projected = project_hard_constraints_latent(
        latent,
        target_latent,
        constraints,
        areas,
        iterations=iterations,
    )
    if straight_through:
        endpoint = latent + strength * (projected - latent).detach()
    else:
        endpoint = torch.lerp(latent, projected, strength)
    return endpoint, {
        "projector_endpoint_delta": (
            endpoint - latent
        ).detach().abs().mean(),
        "projector_endpoint_active_fraction": (
            (endpoint - latent).detach().abs().gt(1.0e-6).float().mean()
        ),
        "projector_endpoint_strength": torch.tensor(
            strength,
            device=latent.device,
            dtype=latent.dtype,
        ),
    }


__all__ = ["projector_aware_endpoint"]
