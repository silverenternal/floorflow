"""Training and rollout helpers for the coupled L0-L3 hierarchical flow."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import torch
import torch.nn.functional as F

from floorflow.chip_flow.hard_constraint_projector import (
    ProjectionOutput,
    compiled_project_hard_constraints_latent,
    project_hard_constraints,
)
from floorflow.chip_flow.head_supervision import (
    smooth_l1_masked,
    structured_hierarchy_flow_loss,
)
from floorflow.chip_flow.hierarchical_advance import (
    advance_hierarchy_state,
    coarse_stability_masks,
)
from floorflow.chip_flow.hierarchical_flow import (
    HierarchicalTokenFlowModel,
    HierarchyPrediction,
    HierarchyState,
    build_hierarchy_targets,
    structural_membership,
)
from floorflow.chip_flow.hierarchy_sampling import sample_hierarchy_flow
from floorflow.chip_flow.model import decode_boxes_batch
from floorflow.chip_flow.projections import project_conditioned_latent
from floorflow.chip_flow.projector_geometry import resolve_projector_config
from floorflow.chip_flow.source_mixing import mix_random_and_shelf_source
from floorflow.chip_flow.sparse_layers import masked_token_mean
from floorflow.chip_flow.structural_relations import structural_relation_features


def hierarchy_source_state(
    latent: torch.Tensor,
    context,
    generator: torch.Generator | None = None,
    *,
    source_mode: str = "random",
    batch: dict | None = None,
) -> HierarchyState:
    """Create a target-free source state from shelf latents and typed masks."""

    batch_size = latent.shape[0]

    def random(shape: tuple[int, ...]) -> torch.Tensor:
        return torch.randn(
            shape,
            device=latent.device,
            dtype=latent.dtype,
            generator=generator,
        )

    if source_mode not in {"random", "derived"}:
        raise ValueError("source_mode must be random or derived")
    if source_mode == "derived":
        if batch is None or context.block_areas is None:
            raise ValueError(
                "derived hierarchy source requires batch inputs and block areas"
            )
        areas = context.block_areas.float()
        scale = areas.sum(dim=1).sqrt().clamp_min(1.0e-6)
        source_boxes = decode_boxes_batch(latent.float(), areas)
        global_state = block_global_state(source_boxes, areas)
        membership = structural_membership(batch["constraints"])
        structural_state = torch.stack(
            [
                masked_token_mean(latent.float(), membership[..., index])
                for index in range(membership.shape[-1])
            ],
            dim=1,
        )
        incidence = context.typed.port_block_incidence.float()
        degree = incidence.sum(dim=1).clamp_min(1.0)
        weighted = torch.bmm(
            incidence.transpose(1, 2),
            latent.float()[..., :2],
        )
        block_center = weighted / degree[..., None]
        port_state = block_center - batch["pins"].float() / scale[:, None, None]
        port_state = port_state * context.typed.port_mask[..., None]
    else:
        global_state = random((batch_size, 4))
        structural_state = (
            random((*context.typed.structural_mask.shape, 3))
            * context.typed.structural_mask[..., None]
        )
        port_state = (
            random((*context.typed.port_tokens.shape[:2], 2))
            * context.typed.port_mask[..., None]
        )
    moment_state = None
    if context.native_group_moment_state:
        if source_mode == "derived":
            moment_state, _ = structural_relation_features(
                source_boxes,
                areas,
                batch["constraints"],
            )
        else:
            moment_state = random(
                (*context.typed.structural_mask.shape, 7)
            ) * context.typed.structural_mask[..., None]
    return HierarchyState(
        global_state=global_state,
        structural_state=structural_state,
        block_state=latent.float(),
        port_state=port_state,
        structural_moment_state=moment_state,
    )


def interpolate_hierarchy(
    source: HierarchyState,
    target: HierarchyState,
    time: torch.Tensor,
) -> HierarchyState:
    global_time = time.view(-1, *([1] * (source.global_state.ndim - 1)))
    structure_time = time.view(
        -1, *([1] * (source.structural_state.ndim - 1))
    )
    block_time = time.view(-1, *([1] * (source.block_state.ndim - 1)))
    port_time = time.view(-1, *([1] * (source.port_state.ndim - 1)))
    return HierarchyState(
        global_state=(
            (1.0 - global_time) * source.global_state
            + global_time * target.global_state
        ),
        structural_state=(
            (1.0 - structure_time) * source.structural_state
            + structure_time * target.structural_state
        ),
        block_state=(
            (1.0 - block_time) * source.block_state
            + block_time * target.block_state
        ),
        port_state=(
            (1.0 - port_time) * source.port_state
            + port_time * target.port_state
        ),
        structural_moment_state=(
            None
            if source.structural_moment_state is None
            or target.structural_moment_state is None
            else (
                (1.0 - structure_time) * source.structural_moment_state
                + structure_time * target.structural_moment_state
            )
        ),
    )


def hierarchy_flow_loss(
    prediction: HierarchyPrediction,
    target: HierarchyState,
    *,
    huber_beta: float,
    global_weight: float,
    structural_weight: float,
    port_weight: float,
    constraints: torch.Tensor | None = None,
    structural_mask: torch.Tensor | None = None,
    port_incidence: torch.Tensor | None = None,
    port_mask: torch.Tensor | None = None,
    structured: bool = False,
    forced_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    def smooth(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if huber_beta > 0.0:
            return F.smooth_l1_loss(value, reference, beta=huber_beta)
        return (value - reference).pow(2).mean()

    if structured:
        if (
            constraints is None
            or structural_mask is None
            or port_incidence is None
            or port_mask is None
        ):
            raise ValueError(
                "structured hierarchy loss requires constraints and head masks"
            )
        terms = structured_hierarchy_flow_loss(
            block_prediction=prediction.block_state,
            block_target=target.block_state,
            global_prediction=prediction.global_state,
            global_target=target.global_state,
            structural_prediction=prediction.structural_state,
            structural_target=target.structural_state,
            port_prediction=prediction.port_state,
            port_target=target.port_state,
            constraints=constraints,
            structural_mask=structural_mask,
            port_incidence=port_incidence,
            port_mask=port_mask,
            huber_beta=huber_beta,
            global_weight=global_weight,
            structural_weight=structural_weight,
            port_weight=port_weight,
            forced_weight=forced_weight,
        )
        structural_moment = (
            torch.zeros((), device=prediction.block_state.device)
            if prediction.structural_moment_state is None
            or target.structural_moment_state is None
            else smooth_l1_masked(
                prediction.structural_moment_state,
                target.structural_moment_state,
                structural_mask,
                beta=huber_beta,
            )
        )
        return (
            terms.total + float(structural_weight) * structural_moment,
            {**terms.as_dict(), "flow_structural_moment": structural_moment},
        )

    block = smooth(prediction.block_state, target.block_state)
    global_level = smooth(prediction.global_state, target.global_state)
    structural = smooth(
        prediction.structural_state,
        target.structural_state,
    )
    port = smooth(prediction.port_state, target.port_state)
    structural_moment = (
        torch.zeros((), device=block.device)
        if prediction.structural_moment_state is None
        or target.structural_moment_state is None
        else smooth_l1_masked(
            prediction.structural_moment_state,
            target.structural_moment_state,
            (
                structural_mask
                if structural_mask is not None
                else torch.ones_like(
                    target.structural_moment_state[..., 0],
                    dtype=torch.bool,
                )
            ),
            beta=huber_beta,
        )
    )
    total = (
        block
        + float(global_weight) * global_level
        + float(structural_weight) * structural
        + float(structural_weight) * structural_moment
        + float(port_weight) * port
    )
    return total, {
        "flow_block": block,
        "flow_global": global_level,
        "flow_structural": structural,
        "flow_structural_moment": structural_moment,
        "flow_port": port,
    }


def block_global_state(
    boxes: torch.Tensor,
    areas: torch.Tensor,
) -> torch.Tensor:
    """Compute the L0 bbox state from decoded block boxes."""

    scale = areas.sum(dim=1).sqrt().clamp_min(1.0e-6)
    x0 = boxes[..., 0].amin(dim=1)
    y0 = boxes[..., 1].amin(dim=1)
    x1 = (boxes[..., 0] + boxes[..., 2]).amax(dim=1)
    y1 = (boxes[..., 1] + boxes[..., 3]).amax(dim=1)
    return torch.stack(
        [
            (x0 + x1) * 0.5 / scale,
            (y0 + y1) * 0.5 / scale,
            (x1 - x0) / scale,
            (y1 - y0) / scale,
        ],
        dim=-1,
    )


def integrate_hierarchy_step(
    model: HierarchicalTokenFlowModel,
    state: HierarchyState,
    time: torch.Tensor,
    context,
    batch: dict,
    *,
    step_size: float,
    projector: bool,
    projector_iterations: int,
    projector_boundary_axes: bool,
    projector_boundary_lock_corners: bool,
    projector_sparse_pairs: bool | None = None,
    projector_strategy: str | None = None,
    projector_grid_size: int | None = None,
    compile_projector: bool = False,
    projection_mode: str = "post_step",
    source_state: HierarchyState | None = None,
    guidance: Callable | None = None,
    collect_projector_diagnostics: bool = True,
    coarse_frozen: bool = False,
    coarse_anchor: bool = False,
    coarse_anchor_tolerance: float = 0.05,
    coarse_anchor_strength: float = 0.25,
    coarse_anchor_max_step: float = 0.1,
    apply_legalizer_action: bool = False,
) -> tuple[HierarchyState, ProjectionOutput | None]:
    """Apply one target-free Euler step with optional hard projection."""

    if projection_mode not in {"post_step", "projection_consistent"}:
        raise ValueError(
            "step projection_mode must be post_step or projection_consistent"
        )
    if projection_mode == "projection_consistent" and not projector:
        raise ValueError("projection_consistent mode requires projector=True")
    if projection_mode == "projection_consistent" and source_state is None:
        raise ValueError("projection_consistent mode requires source_state")
    prediction = model(
        state,
        time,
        context,
        apply_legalizer_action=apply_legalizer_action,
    )
    sparse_pairs, strategy, grid_size = resolve_projector_config(
        model,
        projector_sparse_pairs,
        projector_strategy,
        projector_grid_size,
    )
    denominator = (1.0 - time).clamp_min(1.0e-3)
    anchor_masks = (
        coarse_stability_masks(
            state,
            prediction,
            tolerance=coarse_anchor_tolerance,
            structural_mask=context.typed.structural_mask,
        )
        if coarse_anchor
        else None
    )

    advanced = advance_hierarchy_state(
        state,
        prediction,
        denominator,
        step_size=step_size,
        port_mask=context.typed.port_mask,
        coarse_frozen=coarse_frozen,
        coarse_anchor_masks=anchor_masks,
        coarse_anchor_strength=coarse_anchor_strength,
        coarse_anchor_max_step=coarse_anchor_max_step,
    )
    if guidance is not None:
        advanced = guidance(advanced, batch)
    projection = None
    if projector and projection_mode == "projection_consistent":
        if source_state is None:
            raise ValueError(
                "projection-consistent integration requires a source state"
            )
        if compile_projector:
            projected_prediction = compiled_project_hard_constraints_latent(
                prediction.block_state,
                batch["target_latent"],
                batch["constraints"],
                batch["areas"],
                iterations=projector_iterations,
                boundary_axes_locked=projector_boundary_axes,
                lock_boundary_corners=projector_boundary_lock_corners,
                sparse_pairs=sparse_pairs,
                projection_strategy=strategy,
                grid_size=grid_size,
            )
        else:
            projection = project_hard_constraints(
                prediction.block_state,
                batch["target_latent"],
                batch["constraints"],
                batch["areas"],
                iterations=projector_iterations,
                boundary_axes_locked=projector_boundary_axes,
                lock_boundary_corners=projector_boundary_lock_corners,
                sparse_pairs=sparse_pairs,
                projection_strategy=strategy,
                grid_size=grid_size,
                collect_diagnostics=collect_projector_diagnostics,
            )
            projected_prediction = projection.latent
        next_time = (time + step_size).clamp_max(1.0)
        fraction = next_time.view(
            -1,
            *([1] * (source_state.block_state.ndim - 1)),
        )
        advanced = replace(
            advanced,
            block_state=(
                (1.0 - fraction) * source_state.block_state
                + fraction * projected_prediction
            ),
        )
    elif projector:
        if compile_projector:
            projected_latent = compiled_project_hard_constraints_latent(
                advanced.block_state,
                batch["target_latent"],
                batch["constraints"],
                batch["areas"],
                iterations=projector_iterations,
                boundary_axes_locked=projector_boundary_axes,
                lock_boundary_corners=projector_boundary_lock_corners,
                sparse_pairs=sparse_pairs,
                projection_strategy=strategy,
                grid_size=grid_size,
                collect_diagnostics=collect_projector_diagnostics,
            )
        else:
            projection = project_hard_constraints(
                advanced.block_state,
                batch["target_latent"],
                batch["constraints"],
                batch["areas"],
                iterations=projector_iterations,
                boundary_axes_locked=projector_boundary_axes,
                lock_boundary_corners=projector_boundary_lock_corners,
                sparse_pairs=sparse_pairs,
                projection_strategy=strategy,
                grid_size=grid_size,
            )
            projected_latent = projection.latent
        advanced = replace(advanced, block_state=projected_latent)
    else:
        advanced = replace(
            advanced,
            block_state=project_conditioned_latent(
                advanced.block_state,
                batch["target_latent"],
                batch["constraints"],
            ),
        )
    if model.self_conditioning:
        advanced = replace(
            advanced,
            previous_block_state=prediction.block_state.detach(),
        )
    return advanced, projection


def hierarchical_training_state(
    batch: dict,
    context,
    generator: torch.Generator | None = None,
    *,
    source_mode: str = "random",
    source_noise: float = 0.0,
    source_levels: str = "random",
    source_shelf_probability: float = 0.5,
) -> tuple[HierarchyState, HierarchyState]:
    """Return source and supervised targets on a deployment-aligned path.

    ``shelf`` reuses the exact source block state consumed by sampling.  The
    legacy ``random`` mode is retained for checkpoint reproduction, while
    ``noisy_shelf`` regularizes a deployment-aligned source with Gaussian noise.
    """

    target = build_hierarchy_targets(batch, context)
    if source_mode not in {"random", "shelf", "noisy_shelf", "mixed"}:
        raise ValueError("source_mode must be random, shelf, noisy_shelf or mixed")
    if source_noise < 0.0:
        raise ValueError("source_noise must be non-negative")
    if not 0.0 <= source_shelf_probability <= 1.0:
        raise ValueError("source_shelf_probability must be in [0, 1]")
    if source_mode == "random":
        block_source = torch.randn(
            target.block_state.shape,
            device=target.block_state.device,
            dtype=target.block_state.dtype,
            generator=generator,
        )
    else:
        if "source_latent" not in batch:
            raise ValueError(
                f"{source_mode} source requires batch['source_latent']"
            )
        block_source = batch["source_latent"].to(
            device=target.block_state.device,
            dtype=target.block_state.dtype,
        )
        if block_source.shape != target.block_state.shape:
            raise ValueError(
                "source_latent must match the block-state shape"
            )
        if source_mode == "noisy_shelf" and source_noise > 0.0:
            block_source = block_source + source_noise * torch.randn(
                block_source.shape,
                device=block_source.device,
                dtype=block_source.dtype,
                generator=generator,
            )
        if source_mode == "mixed":
            block_source = mix_random_and_shelf_source(
                block_source,
                generator=generator,
                shelf_probability=source_shelf_probability,
            )
    source = hierarchy_source_state(
        block_source,
        context,
        generator,
        source_mode=source_levels,
        batch=batch,
    )
    return source, target


__all__ = [
    "block_global_state",
    "hierarchical_training_state",
    "hierarchy_flow_loss",
    "hierarchy_source_state",
    "integrate_hierarchy_step",
    "interpolate_hierarchy",
    "sample_hierarchy_flow",
]
