"""Endpoint and short-NFE consistency for the deployed hierarchy source."""

from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn.functional as F


def _advance(
    state: torch.Tensor,
    prediction: torch.Tensor,
    time: torch.Tensor,
    delta: torch.Tensor,
) -> torch.Tensor:
    view = (1.0 - time).clamp_min(1.0e-3)
    view = view.view(-1, *([1] * (state.ndim - 1)))
    return state + (prediction - state) / view * delta.view(
        -1,
        *([1] * (state.ndim - 1)),
    )


def _masked_smooth_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    expanded = mask
    while expanded.ndim < prediction.ndim:
        expanded = expanded.unsqueeze(-1)
    expanded = expanded.expand_as(prediction).float()
    loss = F.smooth_l1_loss(
        prediction.float(),
        target.float(),
        beta=beta,
        reduction="none",
    )
    return (loss * expanded).sum() / expanded.sum().clamp_min(1.0)


def hierarchy_nfe_consistency_loss(
    model,
    state,
    time: torch.Tensor,
    prediction,
    target,
    context,
    args,
    *,
    step: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply an opt-in short integration path with endpoint consistency."""

    weight = float(getattr(args, "nfe_consistency_weight", 0.0))
    if weight <= 0.0:
        return torch.zeros((), device=time.device), {}
    steps = int(getattr(args, "nfe_consistency_steps", 2))
    if steps <= 0:
        raise ValueError("NFE consistency steps must be positive")
    target_weight = float(getattr(args, "nfe_consistency_target_weight", 1.0))
    self_weight = float(getattr(args, "nfe_consistency_self_weight", 0.1))
    if target_weight < 0.0 or self_weight < 0.0:
        raise ValueError("NFE consistency weights must be non-negative")
    beta = float(getattr(args, "flow_huber_beta", 0.1))
    current_state = state
    current_prediction = prediction
    current_time = time.float()
    endpoint_terms = []
    self_terms = []
    for index in range(steps):
        remaining = max(1, steps - index)
        next_time = (
            current_time + (1.0 - current_time) / float(remaining)
        ).clamp_max(0.999)
        delta = next_time - current_time
        next_state = replace(
            current_state,
            global_state=_advance(
                current_state.global_state,
                current_prediction.global_state,
                current_time,
                delta,
            ),
            structural_state=_advance(
                current_state.structural_state,
                current_prediction.structural_state,
                current_time,
                delta,
            ),
            block_state=_advance(
                current_state.block_state,
                current_prediction.block_state,
                current_time,
                delta,
            ),
            port_state=_advance(
                current_state.port_state,
                current_prediction.port_state,
                current_time,
                delta,
            )
            * context.typed.port_mask[..., None],
            structural_moment_state=(
                None
                if current_state.structural_moment_state is None
                or current_prediction.structural_moment_state is None
                else _advance(
                    current_state.structural_moment_state,
                    current_prediction.structural_moment_state,
                    current_time,
                    delta,
                )
            ),
        )
        next_prediction = model(next_state, next_time, context)
        endpoint = (
            _masked_smooth_l1(
                next_prediction.block_state,
                target.block_state,
                torch.ones_like(target.block_state[..., 0], dtype=torch.bool),
                beta,
            )
            + _masked_smooth_l1(
                next_prediction.global_state,
                target.global_state,
                torch.ones_like(target.global_state, dtype=torch.bool),
                beta,
            )
            + _masked_smooth_l1(
                next_prediction.structural_state,
                target.structural_state,
                context.typed.structural_mask,
                beta,
            )
            + _masked_smooth_l1(
                next_prediction.port_state,
                target.port_state,
                context.typed.port_mask,
                beta,
            )
        )
        if (
            next_prediction.structural_moment_state is not None
            and target.structural_moment_state is not None
        ):
            endpoint = endpoint + _masked_smooth_l1(
                next_prediction.structural_moment_state,
                target.structural_moment_state,
                context.typed.structural_mask,
                beta,
            )
        consistent = _masked_smooth_l1(
            next_prediction.block_state,
            current_prediction.block_state.detach(),
            torch.ones_like(target.block_state[..., 0], dtype=torch.bool),
            beta,
        )
        endpoint_terms.append(endpoint)
        self_terms.append(consistent)
        current_state = next_state
        current_prediction = next_prediction
        current_time = next_time
    endpoint = torch.stack(endpoint_terms).mean()
    consistent = torch.stack(self_terms).mean()
    total = weight * (
        target_weight * endpoint + self_weight * consistent
    )
    return total, {
        "nfe_consistency": total,
        "nfe_consistency_endpoint": endpoint,
        "nfe_consistency_self": consistent,
        "nfe_consistency_current_weight": total.new_tensor(weight),
        "nfe_consistency_step": total.new_tensor(float(step)),
    }


__all__ = ["hierarchy_nfe_consistency_loss"]
