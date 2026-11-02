"""Conflict-aware backward for competing floorplan objectives."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def conflict_aware_backward(
    losses: dict[str, torch.Tensor],
    parameters: Sequence[torch.nn.Parameter],
    conflict_pairs: Sequence[tuple[str, str]],
    *,
    loss_scale: float = 1.0,
    total_loss: torch.Tensor | None = None,
) -> dict[str, float]:
    """Project primary gradients away from conflicting objectives, then combine."""

    if not losses:
        raise ValueError("at least one loss is required")
    if loss_scale <= 0.0:
        raise ValueError("loss_scale must be positive")
    names = tuple(losses)
    if total_loss is not None:
        return _incremental_conflict_backward(
            names,
            losses,
            parameters,
            conflict_pairs,
            loss_scale=float(loss_scale),
            total_loss=total_loss,
        )
    gradients = {}
    for name, loss in losses.items():
        values = torch.autograd.grad(
            loss * float(loss_scale),
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        gradients[name] = torch.cat(
            [
                (
                    torch.zeros_like(parameter).flatten()
                    if gradient is None
                    else gradient.detach().flatten()
                )
                for parameter, gradient in zip(parameters, values)
            ]
        )
    projected = {name: value.clone() for name, value in gradients.items()}
    stat_values = {}
    for primary, conflict in conflict_pairs:
        if primary not in projected or conflict not in projected:
            raise ValueError(f"unknown conflict pair: {primary}/{conflict}")
        primary_gradient = projected[primary]
        conflict_gradient = gradients[conflict]
        denominator = conflict_gradient.square().sum().clamp_min(1.0e-12)
        dot = torch.dot(primary_gradient, conflict_gradient)
        cosine = dot / (
            primary_gradient.norm() * conflict_gradient.norm()
        ).clamp_min(1.0e-12)
        did_project = dot < 0
        projected_value = primary_gradient - (
            dot / denominator
        ) * conflict_gradient
        projected[primary] = torch.where(
            did_project,
            projected_value,
            primary_gradient,
        )
        stat_values[f"{primary}__{conflict}__cosine"] = cosine
        stat_values[f"{primary}__{conflict}__projected"] = (
            did_project.to(cosine.dtype)
        )
    combined = torch.stack([projected[name] for name in names]).sum(dim=0)
    offset = 0
    for parameter in parameters:
        length = parameter.numel()
        parameter.grad = combined[offset : offset + length].view_as(parameter)
        offset += length
    stat_values["combined_gradient_norm"] = combined.norm() / loss_scale
    names = tuple(stat_values)
    values = torch.stack([stat_values[name] for name in names]).cpu().tolist()
    return dict(zip(names, values))


def _incremental_conflict_backward(
    names: tuple[str, ...],
    losses: dict[str, torch.Tensor],
    parameters: Sequence[torch.nn.Parameter],
    conflict_pairs: Sequence[tuple[str, str]],
    *,
    loss_scale: float,
    total_loss: torch.Tensor,
) -> dict[str, float]:
    """Preserve the full objective and backpropagate only pair corrections."""

    required = {
        name
        for pair in conflict_pairs
        for name in pair
    }
    for primary, conflict in conflict_pairs:
        if primary not in losses or conflict not in losses:
            raise ValueError(f"unknown conflict pair: {primary}/{conflict}")
    total_loss.backward(retain_graph=True)
    gradients = {}
    for name in sorted(required):
        values = torch.autograd.grad(
            losses[name] * loss_scale,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        gradients[name] = torch.cat(
            [
                (
                    torch.zeros_like(parameter).flatten()
                    if gradient is None
                    else gradient.detach().flatten()
                )
                for parameter, gradient in zip(parameters, values)
            ]
        )
    current_primary = {
        name: gradients[name].clone()
        for name in required
    }
    corrections: dict[str, torch.Tensor] = {
        name: torch.zeros_like(gradients[name])
        for name in required
    }
    stat_values = {}
    for primary, conflict in conflict_pairs:
        primary_gradient = current_primary[primary]
        conflict_gradient = gradients[conflict]
        denominator = conflict_gradient.square().sum().clamp_min(1.0e-12)
        dot = torch.dot(primary_gradient, conflict_gradient)
        cosine = dot / (
            primary_gradient.norm() * conflict_gradient.norm()
        ).clamp_min(1.0e-12)
        did_project = dot < 0
        projected_value = primary_gradient - (
            dot / denominator
        ) * conflict_gradient
        updated = torch.where(
            did_project,
            projected_value,
            primary_gradient,
        )
        current_primary[primary] = updated
        corrections[primary] = updated - gradients[primary]
        stat_values[f"{primary}__{conflict}__cosine"] = cosine
        stat_values[f"{primary}__{conflict}__projected"] = (
            did_project.to(cosine.dtype)
        )
    total_correction = torch.stack(
        [corrections[name] for name in names if name in corrections]
    ).sum(dim=0)
    offset = 0
    for parameter in parameters:
        length = parameter.numel()
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        parameter.grad.add_(
            total_correction[offset : offset + length].view_as(parameter)
        )
        offset += length
    combined = torch.cat(
        [parameter.grad.detach().flatten() for parameter in parameters]
    )
    stat_values["combined_gradient_norm"] = combined.norm() / loss_scale
    names = tuple(stat_values)
    values = torch.stack([stat_values[name] for name in names]).cpu().tolist()
    return dict(zip(names, values))


__all__ = ["conflict_aware_backward"]
