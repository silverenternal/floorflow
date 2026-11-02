"""Inference-time structural energy guidance for hierarchical flow sampling."""

from __future__ import annotations

from dataclasses import replace

import torch

from floorflow.chip_flow.batched import (
    batch_grouping_proxy,
    batch_physics_terms,
    batch_routing_proxy_loss,
)
from floorflow.chip_flow.hierarchical_flow import HierarchicalTokenFlowModel
from floorflow.chip_flow.hierarchical_flow_runtime import (
    hierarchy_source_state,
    integrate_hierarchy_step,
)
from floorflow.chip_flow.model import decode_boxes_batch


def sample_hierarchy_guided_flow(
    model: HierarchicalTokenFlowModel,
    batch: dict,
    *,
    sampling_steps: int,
    generator: torch.Generator,
    guidance_scale: float,
    energy_mode: str = "routing",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply normalized routing-proxy gradients between Euler steps."""

    if guidance_scale < 0.0:
        raise ValueError("guidance_scale must be non-negative")
    if energy_mode not in {"routing", "composite"}:
        raise ValueError("energy_mode must be routing or composite")
    context = model.prepare_context(batch)
    state = hierarchy_source_state(batch["source_latent"], context, generator)
    energies = []
    gradient_rms = []
    update_rms = []

    def guide(current, current_batch):
        with torch.enable_grad():
            latent = current.block_state.detach().requires_grad_(True)
            boxes = decode_boxes_batch(latent, current_batch["areas"])
            energy = batch_routing_proxy_loss(
                boxes,
                current_batch["areas"],
                current_batch["b2b"],
                current_batch["p2b"],
                current_batch["pins"],
                current_batch["n_b2b"],
                current_batch["n_p2b"],
                current_batch["n_pins"],
            ).sum()
            if energy_mode == "composite":
                grouping = batch_grouping_proxy(
                    boxes,
                    current_batch["constraints"],
                ).sum()
                physics = batch_physics_terms(
                    boxes,
                    current_batch["areas"],
                    current_batch["constraints"],
                    current_batch["b2b"],
                    current_batch["p2b"],
                    current_batch["pins"],
                    current_batch["n_b2b"],
                    current_batch["n_p2b"],
                    current_batch["n_pins"],
                )
                energy = (
                    0.25 * energy
                    + grouping
                    + physics["overlap"].sum()
                    + 0.5 * physics["boundary"].sum()
                    + 0.1 * physics["wl"].sum()
                    + 0.1 * physics["area"].sum()
                )
            gradient = torch.autograd.grad(energy, latent)[0]
            rms = gradient.square().mean(dim=(1, 2)).sqrt().clamp_min(1.0e-8)
            correction = (
                guidance_scale
                * gradient
                / rms[:, None, None]
                / max(1, sampling_steps)
            )
            energies.append(float(energy.detach()) / latent.shape[0])
            gradient_rms.append(float(rms.mean()))
            update_rms.append(
                float(correction.square().mean(dim=(1, 2)).sqrt().mean())
            )
        return replace(current, block_state=(latent - correction).detach())

    with torch.no_grad():
        for step in range(sampling_steps):
            time = torch.full(
                (state.block_state.shape[0],),
                step / sampling_steps,
                device=state.block_state.device,
            )
            state, _ = integrate_hierarchy_step(
                model,
                state,
                time,
                context,
                batch,
                step_size=1.0 / sampling_steps,
                projector=model.projector,
                projector_iterations=model.projector_iterations,
                projector_boundary_axes=model.projector_boundary_axes,
                projector_boundary_lock_corners=model.projector_boundary_lock_corners,
                guidance=guide,
            )
    diagnostics = {
        "guidance_energy_mean": torch.tensor(
            sum(energies) / max(1, len(energies)),
            device=state.block_state.device,
        ),
        "guidance_gradient_rms_mean": torch.tensor(
            sum(gradient_rms) / max(1, len(gradient_rms)),
            device=state.block_state.device,
        ),
        "guidance_update_rms_mean": torch.tensor(
            sum(update_rms) / max(1, len(update_rms)),
            device=state.block_state.device,
        ),
    }
    return state.block_state, diagnostics


__all__ = ["sample_hierarchy_guided_flow"]
