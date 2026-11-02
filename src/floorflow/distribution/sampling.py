"""Sampling helpers for contour-initialized distribution CFM."""

from __future__ import annotations

import torch


@torch.inference_mode()
def sample_distribution_floorplans(
    model,
    footprint: torch.Tensor,
    *,
    floor_ids: torch.Tensor | None = None,
    style_ids: torch.Tensor | None = None,
    steps: int = 4,
    noise_scale: float = 0.0,
    style_gate: float = 1.0,
    solver: str = "euler",
    source_mode: str = "blend",
) -> tuple[torch.Tensor, dict]:
    """Integrate from the same contour-derived source used during training."""

    if footprint.ndim == 3:
        footprint = footprint[:, None]
    condition_features = (
        model.encode_condition(footprint, floor_ids=floor_ids)
        if hasattr(model, "encode_condition")
        else None
    )
    initial_kwargs = {
        "floor_ids": floor_ids,
        "style_ids": style_ids,
    }
    if condition_features is not None:
        initial_kwargs["condition_features"] = condition_features
    initial = model.contour_initial_state(footprint, **initial_kwargs)
    if source_mode not in {"blend", "pixel", "element"}:
        raise ValueError("source_mode must be blend, pixel, or element")
    source_key = "state" if source_mode == "blend" else f"{source_mode}_state"
    state = initial["cfm"][source_key].clone()
    gate = min(1.0, max(0.0, float(style_gate)))
    if style_ids is not None and gate < 1.0:
        base_kwargs = {"floor_ids": floor_ids, "style_ids": None}
        if condition_features is not None:
            base_kwargs["condition_features"] = condition_features
        base_initial = model.contour_initial_state(
            footprint,
            **base_kwargs,
        )
        state = base_initial["cfm"][source_key] + gate * (
            state - base_initial["cfm"][source_key]
        )
    if noise_scale > 0:
        state = (
            state + float(noise_scale) * torch.randn_like(state)
        ) * footprint.float()
    steps = max(0, int(steps))
    if steps == 0:
        return state, initial
    dt = 1.0 / steps

    def velocity_at(state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        model_kwargs = {
            "floor_ids": floor_ids,
            "style_ids": style_ids,
            "source": initial["cfm"][source_key],
        }
        if condition_features is not None:
            model_kwargs["condition_features"] = condition_features
        velocity = model(
            state,
            time,
            footprint,
            **model_kwargs,
        )
        if style_ids is not None and gate < 1.0:
            base_kwargs = {
                "floor_ids": floor_ids,
                "style_ids": None,
                "source": initial["cfm"][source_key],
            }
            if condition_features is not None:
                base_kwargs["condition_features"] = condition_features
            base_velocity = model(
                state,
                time,
                footprint,
                **base_kwargs,
            )
            velocity = base_velocity + gate * (velocity - base_velocity)
        return velocity

    if solver not in {"euler", "shifted_euler", "heun"}:
        raise ValueError("solver must be euler, shifted_euler, or heun")
    for step in range(steps):
        time_offset = 0.5 if solver == "shifted_euler" else 0.0
        time = torch.full(
            (footprint.shape[0],),
            (step + time_offset) * dt,
            device=footprint.device,
            dtype=footprint.dtype,
        )
        first_velocity = velocity_at(state, time)
        if solver == "heun":
            predictor = (state + dt * first_velocity) * footprint.float()
            next_time = torch.full_like(time, min(1.0, (step + 1) * dt))
            second_velocity = velocity_at(predictor, next_time)
            update = 0.5 * (first_velocity + second_velocity)
        else:
            update = first_velocity
        state = (state + dt * update) * footprint.float()
    return state, initial


__all__ = ["sample_distribution_floorplans"]
