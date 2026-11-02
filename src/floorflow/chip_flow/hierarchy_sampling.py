"""Target-free sampling for the coupled L0-L3 chip hierarchy."""

from __future__ import annotations

from dataclasses import replace

import torch

from floorflow.chip_flow.hard_constraint_projector import project_hard_constraints
from floorflow.chip_flow.hierarchical_advance import coarse_commit_threshold
from floorflow.chip_flow.projector_geometry import resolve_projector_config
from floorflow.chip_flow.shape_mode_features import (
    select_pareto_safe_shape_mode,
    shape_mode_geometry_features,
)


@torch.no_grad()
def sample_hierarchy_flow(
    model,
    batch: dict,
    *,
    sampling_steps: int,
    generator: torch.Generator,
    projector: bool,
    projector_iterations: int,
    projector_boundary_axes: bool = False,
    projector_boundary_lock_corners: bool = True,
    projector_sparse_pairs: bool | None = None,
    projector_strategy: str | None = None,
    projector_grid_size: int | None = None,
    compile_projector: bool = False,
    projection_mode: str = "post_step",
    coarse_commit_fraction: float | None = None,
    coarse_anchor_tolerance: float = 0.05,
    coarse_anchor_strength: float = 0.25,
    coarse_anchor_max_step: float = 0.1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Sample a hierarchy trajectory with optional bounded coarse anchoring."""

    from floorflow.chip_flow.hierarchical_flow_runtime import (
        hierarchy_source_state,
        integrate_hierarchy_step,
    )

    if projection_mode not in {"post_step", "projection_consistent", "final_only"}:
        raise ValueError(f"invalid projection_mode: {projection_mode}")
    if projection_mode != "post_step" and not projector:
        raise ValueError(f"{projection_mode} mode requires projector=True")
    if coarse_anchor_tolerance < 0.0:
        raise ValueError("coarse anchor tolerance must be non-negative")
    if not 0.0 <= coarse_anchor_strength <= 1.0:
        raise ValueError("coarse anchor strength must be in [0, 1]")
    if coarse_anchor_max_step < 0.0:
        raise ValueError("coarse anchor max step must be non-negative")
    context = model.prepare_context(batch)
    source_state = hierarchy_source_state(
        batch["source_latent"],
        context,
        generator,
        source_mode=getattr(model, "hierarchy_source_levels", "random"),
        batch=batch,
    )
    state = source_state
    commit_step = (
        coarse_commit_threshold(coarse_commit_fraction, sampling_steps)
        if coarse_commit_fraction is not None else sampling_steps
    )
    residual_overlap = boundary_violations = mib_violations = 0
    for step in range(sampling_steps):
        time = torch.full(
            (state.block_state.shape[0],),
            step / sampling_steps,
            device=state.block_state.device,
        )
        use_bounded_anchor = (
            step >= commit_step and coarse_anchor_strength > 0.0
        )
        state, projection = integrate_hierarchy_step(
            model,
            state,
            time,
            context,
            batch,
            step_size=1.0 / sampling_steps,
            projector=projector and projection_mode != "final_only",
            projector_iterations=projector_iterations,
            projector_boundary_axes=projector_boundary_axes,
            projector_boundary_lock_corners=projector_boundary_lock_corners,
            projector_sparse_pairs=projector_sparse_pairs,
            projector_strategy=projector_strategy,
            projector_grid_size=projector_grid_size,
            compile_projector=compile_projector,
            projection_mode=(
                "projection_consistent"
                if projection_mode == "projection_consistent"
                else "post_step"
            ),
            source_state=source_state,
            collect_projector_diagnostics=step == sampling_steps - 1,
            coarse_frozen=step >= commit_step and not use_bounded_anchor,
            coarse_anchor=use_bounded_anchor,
            coarse_anchor_tolerance=coarse_anchor_tolerance,
            coarse_anchor_strength=coarse_anchor_strength,
            coarse_anchor_max_step=coarse_anchor_max_step,
            apply_legalizer_action=step == sampling_steps - 1,
        )
        if projection is not None:
            residual_overlap += projection.remaining_overlap_pairs
            boundary_violations += projection.boundary_violations
            mib_violations += projection.mib_shape_violations
    if projection_mode == "final_only":
        sparse_pairs, strategy, grid_size = resolve_projector_config(
            model,
            projector_sparse_pairs,
            projector_strategy,
            projector_grid_size,
        )
        projection = project_hard_constraints(
            state.block_state,
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
        state = replace(state, block_state=projection.latent)
        residual_overlap += projection.remaining_overlap_pairs
        boundary_violations += projection.boundary_violations
        mib_violations += projection.mib_shape_violations
    shape_diagnostics = {}
    if getattr(model, "shape_mode_head", None) is not None:
        shape_output = model.shape_mode_head.forward_with_latents(
            state.block_state,
            context.typed.block_tokens,
            state.block_state,
            constraints=context.constraints,
            membership=context.coarse_membership,
            structural_tokens=context.typed.structural_tokens,
            structural_mask=context.typed.structural_mask,
            global_token=context.typed.global_token[:, 0],
        )
        candidate_features = shape_mode_geometry_features(
            shape_output.mode_latents,
            state.block_state,
            batch,
            include_baseline=True,
        )
        mode_features = candidate_features[:, 1:]
        shape_output = model.shape_mode_head.forward_with_latents(
            state.block_state,
            context.typed.block_tokens,
            state.block_state,
            constraints=context.constraints,
            membership=context.coarse_membership,
            structural_tokens=context.typed.structural_tokens,
            structural_mask=context.typed.structural_mask,
            global_token=context.typed.global_token[:, 0],
            mode_features=mode_features,
        )
        candidate_latents = torch.cat(
            [state.block_state[:, None], shape_output.mode_latents],
            dim=1,
        )
        candidate_logits = None
        if getattr(model, "shape_mode_selection", "learned") == "pareto":
            selected_candidate = select_pareto_safe_shape_mode(
                candidate_features
            )
        else:
            baseline_logit = model.shape_mode_head.baseline_logit(
                candidate_features[:, 0]
            )
            candidate_logits = torch.cat(
                [baseline_logit[:, None], shape_output.mode_logits],
                dim=-1,
            )
            selected_candidate = candidate_logits.argmax(dim=-1)
        selected_mode = selected_candidate - 1
        selected_latent = candidate_latents.gather(
            1,
            selected_candidate[:, None, None, None].expand(
                -1,
                -1,
                state.block_state.shape[1],
                state.block_state.shape[2],
            ),
        )[:, 0]
        if getattr(model, "shape_mode_apply", False):
            state = replace(state, block_state=selected_latent)
            mode_selected = selected_candidate.gt(0)
            if projector and bool(mode_selected.any()):
                sparse_pairs, strategy, grid_size = resolve_projector_config(
                    model,
                    projector_sparse_pairs,
                    projector_strategy,
                    projector_grid_size,
                )
                shape_projection = project_hard_constraints(
                    state.block_state,
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
                state = replace(
                    state,
                    block_state=torch.where(
                        mode_selected[:, None, None],
                        shape_projection.latent,
                        state.block_state,
                    ),
                )
        shape_diagnostics = {
            "shape_mode_latents": shape_output.mode_latents,
            "shape_mode_rho_residuals": shape_output.rho_residuals,
            "shape_mode_center_residuals": shape_output.center_residuals,
            "shape_mode_logits": shape_output.mode_logits,
            "shape_mode_candidate_logits": candidate_logits,
            "shape_mode_selected": selected_mode,
            "shape_mode_selected_candidate": selected_candidate,
            "shape_mode_selected_latent": selected_latent,
            "shape_mode_geometry_features": candidate_features,
        }
        if candidate_logits is None:
            shape_diagnostics.pop("shape_mode_candidate_logits")
        if (
            getattr(model, "shape_mode_apply", False)
            and projector
        ):
            shape_diagnostics["shape_mode_projected"] = (
                state.block_state.new_ones(())
            )
    if not projector:
        return state.block_state, shape_diagnostics
    cases = state.block_state.shape[0]
    if compile_projector:
        return state.block_state, {
            **shape_diagnostics,
            "projector_residual_overlap_mean": torch.zeros(
                (),
                device=state.block_state.device,
            ),
            "projector_boundary_violations_mean": torch.zeros(
                (),
                device=state.block_state.device,
            ),
            "projector_mib_violations_mean": torch.zeros(
                (),
                device=state.block_state.device,
            ),
            "projector_diagnostics_mode": torch.zeros(
                (),
                device=state.block_state.device,
            ),
        }
    return state.block_state, {
        **shape_diagnostics,
        "projector_residual_overlap_mean": torch.tensor(
            residual_overlap / cases,
            device=state.block_state.device,
        ),
        "projector_boundary_violations_mean": torch.tensor(
            boundary_violations / cases,
            device=state.block_state.device,
        ),
        "projector_mib_violations_mean": torch.tensor(
            mib_violations / cases,
            device=state.block_state.device,
        ),
    }


__all__ = ["sample_hierarchy_flow"]
