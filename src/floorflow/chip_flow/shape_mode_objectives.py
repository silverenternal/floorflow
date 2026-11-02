"""Objectives for bounded chip shape-mode candidates."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from floorflow.chip_flow.hierarchy_state import HierarchyPrediction
from floorflow.chip_flow.projections import project_conditioned_latent
from floorflow.chip_flow.shape_mode_features import (
    shape_mode_geometry_features,
)


@dataclass(frozen=True)
class ShapeModeOutput:
    """Mode-specific endpoint residuals and their gated model output."""

    rho_residuals: torch.Tensor
    center_residuals: torch.Tensor
    mode_latents: torch.Tensor
    mode_logits: torch.Tensor


def winner_take_all_loss(
    mode_latents: torch.Tensor,
    target_latent: torch.Tensor,
    *,
    beta: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return best-mode smooth-L1 loss and winner indices."""

    if mode_latents.ndim != 4 or target_latent.ndim != 3:
        raise ValueError("mode/target latent ranks must be four and three")
    if mode_latents.shape[0] != target_latent.shape[0] or (
        mode_latents.shape[2:] != target_latent.shape[1:]
    ):
        raise ValueError("mode and target latent shapes must match")
    per_mode = F.smooth_l1_loss(
        mode_latents.float(),
        target_latent.float()[:, None].expand_as(mode_latents),
        beta=beta,
        reduction="none",
    ).mean(dim=(2, 3))
    best_loss, winner = per_mode.min(dim=1)
    return best_loss.mean(), winner


def mode_diversity_loss(
    mode_rho: torch.Tensor,
    *,
    margin: float = 0.05,
) -> torch.Tensor:
    """Encourage pairwise mode separation up to a bounded margin."""

    if mode_rho.ndim != 3 or mode_rho.shape[1] < 2:
        raise ValueError("mode_rho must have shape (batch, modes>=2, blocks)")
    if margin < 0.0:
        raise ValueError("diversity margin must be non-negative")
    distance = (mode_rho[:, :, None] - mode_rho[:, None, :]).abs()
    mask = ~torch.eye(
        mode_rho.shape[1],
        dtype=torch.bool,
        device=mode_rho.device,
    )[None]
    penalty = F.relu(float(margin) - distance) * mask[..., None]
    return penalty.sum() / mask.sum().clamp_min(1) / mode_rho.shape[2]


def mode_usage_loss(
    mode_losses: torch.Tensor,
    *,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Penalize collapsed soft assignment over shape modes."""

    if mode_losses.ndim != 2 or mode_losses.shape[1] < 2:
        raise ValueError("mode_losses must have shape (batch, modes>=2)")
    if temperature <= 0.0:
        raise ValueError("usage temperature must be positive")
    assignment = torch.softmax(
        -mode_losses.float() / float(temperature),
        dim=-1,
    )
    usage = assignment.mean(dim=0)
    entropy = -(usage.clamp_min(1.0e-8).log() * usage).sum()
    return entropy.new_tensor(float(mode_losses.shape[1])).log() - entropy


def shape_mode_auxiliary_loss(
    mode_latents: torch.Tensor,
    target_latent: torch.Tensor,
    *,
    mode_logits: torch.Tensor | None = None,
    baseline_latent: torch.Tensor | None = None,
    baseline_logit: torch.Tensor | None = None,
    generator_weight: float = 0.1,
    diversity_weight: float = 0.01,
    usage_weight: float = 0.01,
    selection_weight: float = 0.1,
    diversity_margin: float = 0.05,
    usage_temperature: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Combine WTA endpoint, mode diversity and usage balancing."""

    candidate_latents = mode_latents
    candidate_logits = mode_logits
    if baseline_latent is not None or baseline_logit is not None:
        if baseline_latent is None or baseline_logit is None:
            raise ValueError(
                "baseline_latent and baseline_logit must be provided together"
            )
        candidate_latents = torch.cat(
            [baseline_latent[:, None], mode_latents],
            dim=1,
        )
        if mode_logits is not None:
            candidate_logits = torch.cat(
                [baseline_logit[:, None], mode_logits],
                dim=-1,
            )
    wta, winner = winner_take_all_loss(
        candidate_latents,
        target_latent,
    )
    generator, _generator_winner = winner_take_all_loss(
        mode_latents,
        target_latent,
    )
    diversity = mode_diversity_loss(
        mode_latents.flatten(start_dim=2),
        margin=diversity_margin,
    )
    per_mode = F.smooth_l1_loss(
        candidate_latents.float(),
        target_latent.float()[:, None].expand_as(candidate_latents),
        beta=0.1,
        reduction="none",
    ).mean(dim=(2, 3))
    usage = mode_usage_loss(per_mode, temperature=usage_temperature)
    selection = wta.new_zeros(())
    if candidate_logits is not None and float(selection_weight) > 0.0:
        if candidate_logits.shape != per_mode.shape:
            raise ValueError("shape mode logits must match [B,modes]")
        selection = F.cross_entropy(candidate_logits.float(), winner)
    total = (
        wta
        + float(generator_weight) * generator
        + float(diversity_weight) * diversity
        + float(usage_weight) * usage
        + float(selection_weight) * selection
    )
    return {
        "total": total,
        "wta": wta,
        "generator": generator,
        "diversity": diversity,
        "usage": usage,
        "selection": selection,
        "winner": winner,
    }


def shape_mode_training_loss(
    prediction: HierarchyPrediction,
    target,
    args,
    runtime: tuple[object, ...] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return the opt-in training loss and scalar diagnostics."""

    weight = float(getattr(args, "shape_mode_loss_weight", 0.0))
    mode_latents = prediction.shape_mode_latents
    mode_logits = prediction.shape_mode_logits
    baseline_logit = None
    baseline_state = prediction.block_state
    if mode_latents is None and runtime is not None:
        if len(runtime) < 4:
            raise ValueError(
                "shape-mode training requires model, state, context and batch"
            )
        model, state, context = runtime[:3]
        batch = runtime[3]
        baseline_state = project_conditioned_latent(
            prediction.block_state,
            batch["target_latent"],
            context.constraints,
        )
        output = shape_mode_output_from_prediction(
            model,
            prediction,
            state,
            context,
            baseline_state=baseline_state,
        )
        if output is not None:
            mode_latents = output.mode_latents
            mode_logits = output.mode_logits
            candidate_features = shape_mode_geometry_features(
                output.mode_latents,
                baseline_state,
                batch,
                include_baseline=True,
            )
            mode_features = candidate_features[:, 1:]
            typed = context.typed
            mode_logits = model.shape_mode_head.forward_with_latents(
                baseline_state,
                typed.block_tokens,
                state.block_state,
                constraints=context.constraints,
                membership=context.coarse_membership,
                structural_tokens=typed.structural_tokens,
                structural_mask=typed.structural_mask,
                global_token=typed.global_token[:, 0],
                mode_features=mode_features,
            ).mode_logits
            baseline_logit = model.shape_mode_head.baseline_logit(
                candidate_features[:, 0]
            )
    if mode_latents is None or weight <= 0.0:
        device = prediction.block_state.device
        return torch.zeros((), device=device), {}
    terms = shape_mode_auxiliary_loss(
        mode_latents,
        target.block_state,
        mode_logits=mode_logits,
        baseline_latent=baseline_state if baseline_logit is not None else None,
        baseline_logit=baseline_logit,
        generator_weight=float(
            getattr(args, "shape_mode_generator_weight", 0.1)
        ),
        diversity_weight=float(
            getattr(args, "shape_mode_diversity_weight", 0.01)
        ),
        usage_weight=float(getattr(args, "shape_mode_usage_weight", 0.01)),
        selection_weight=float(
            getattr(args, "shape_mode_selection_weight", 0.1)
        ),
        diversity_margin=float(
            getattr(args, "shape_mode_diversity_margin", 0.05)
        ),
        usage_temperature=float(
            getattr(args, "shape_mode_usage_temperature", 0.1)
        ),
    )
    return weight * terms["total"], {
        "shape_mode": weight * terms["total"],
        "shape_mode_wta": terms["wta"],
        "shape_mode_generator": terms["generator"],
        "shape_mode_diversity": terms["diversity"],
        "shape_mode_usage": terms["usage"],
        "shape_mode_selection": terms["selection"],
        "shape_mode_current_weight": prediction.block_state.new_tensor(weight),
    }


def shape_mode_output_from_prediction(
    model: nn.Module,
    prediction: HierarchyPrediction,
    state,
    context,
    *,
    baseline_state: torch.Tensor | None = None,
) -> ShapeModeOutput | None:
    """Generate endpoint shape candidates outside the solver hot loop."""

    head = getattr(model, "shape_mode_head", None)
    if head is None:
        return None
    typed = context.typed
    return head.forward_with_latents(
        prediction.block_state if baseline_state is None else baseline_state,
        typed.block_tokens,
        state.block_state,
        membership=context.coarse_membership,
        structural_tokens=typed.structural_tokens,
        structural_mask=typed.structural_mask,
        global_token=typed.global_token[:, 0],
    )


def shape_mode_latents_from_prediction(
    model: nn.Module,
    prediction: HierarchyPrediction,
    state,
    context,
    *,
    baseline_state: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Generate endpoint shape candidates outside the solver hot loop."""

    output = shape_mode_output_from_prediction(
        model,
        prediction,
        state,
        context,
        baseline_state=baseline_state,
    )
    return None if output is None else output.mode_latents


__all__ = [
    "ShapeModeOutput",
    "mode_diversity_loss",
    "mode_usage_loss",
    "shape_mode_auxiliary_loss",
    "shape_mode_latents_from_prediction",
    "shape_mode_output_from_prediction",
    "shape_mode_training_loss",
    "winner_take_all_loss",
]
