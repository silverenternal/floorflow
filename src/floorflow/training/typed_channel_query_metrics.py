"""Query-level diagnostics for typed-channel validation."""

from __future__ import annotations

import torch
from scipy.optimize import linear_sum_assignment

from floorflow.objectives.room_query_matching import match_room_mask_queries
from floorflow.training.program_instance_training import (
    program_instance_assignment_loss,
)


def multiclass_ece(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    bins: int = 10,
) -> torch.Tensor:
    probability = logits.float().softmax(dim=-1)
    confidence, prediction = probability.max(dim=-1)
    correct = prediction.eq(target).float()
    ece = logits.new_zeros(logits.shape[0])
    for lower in torch.linspace(
        0.0,
        1.0,
        bins + 1,
        device=logits.device,
    )[:-1]:
        upper = lower + 1.0 / bins
        selected = (confidence >= lower) & (
            (confidence < upper) if upper < 1.0 else (confidence <= upper)
        )
        if selected.any():
            gap = (
                correct[selected].mean() - confidence[selected].mean()
            ).abs()
            ece[selected] = gap
    return ece


def poisson_binomial_count_probability(
    existence_logits: torch.Tensor,
) -> torch.Tensor:
    """Convert independent query-presence probabilities into a count PMF."""

    probability = existence_logits.float().sigmoid()
    distribution = existence_logits.new_ones(
        (existence_logits.shape[0], 1),
        dtype=torch.float32,
    )
    for query in range(probability.shape[1]):
        presence = probability[:, query : query + 1]
        updated = existence_logits.new_zeros(
            (existence_logits.shape[0], distribution.shape[1] + 1),
            dtype=torch.float32,
        )
        updated[:, :-1] += distribution * (1.0 - presence)
        updated[:, 1:] += distribution * presence
        distribution = updated
    return distribution


def program_assignment_metrics(
    output: dict[str, torch.Tensor],
    index: int,
    class_logits: torch.Tensor,
    mask_logits: torch.Tensor,
    target_classes: torch.Tensor,
    target_masks: torch.Tensor,
    target_valid: torch.Tensor,
    program: torch.Tensor,
) -> dict[str, float]:
    assignment = output.get("program_room_assignment_logits")
    if assignment is None:
        return {"accuracy": 0.0, "loss": 0.0}
    matches = match_room_mask_queries(
        class_logits,
        mask_logits,
        target_classes,
        target_masks,
        target_valid,
    )
    loss, components = program_instance_assignment_loss(
        assignment[index : index + 1],
        mask_logits,
        target_classes,
        target_masks,
        target_valid,
        program,
        matches=matches,
    )
    return {
        "accuracy": float(
            components["program_instance_assignment_accuracy"]
        ),
        "loss": float(loss),
    }


def soft_query_mask_iou(
    mask_logits: torch.Tensor,
    target_masks: torch.Tensor,
    target_valid: torch.Tensor,
) -> float:
    probability = mask_logits[0].sigmoid().flatten(1)
    targets = target_masks[0].flatten(1)
    count = int(target_valid[0].sum())
    if count == 0:
        return 0.0
    intersection = probability @ targets[:count].T
    union = (
        probability.sum(dim=-1)[:, None]
        + targets[:count].sum(dim=-1)[None]
        - intersection
    )
    similarity = intersection / union.clamp_min(1.0e-8)
    rows, columns = linear_sum_assignment(
        -similarity.detach().float().cpu().numpy()
    )
    return float(similarity[rows, columns].mean())


def matched_query_class_metrics(
    mask_logits: torch.Tensor,
    class_logits: torch.Tensor,
    target_classes: torch.Tensor,
    target_masks: torch.Tensor,
    target_valid: torch.Tensor,
) -> dict[str, float]:
    probability = mask_logits[0].sigmoid().flatten(1)
    targets = target_masks[0].flatten(1)
    count = int(target_valid[0].sum())
    if count == 0:
        return {"nll": 0.0, "margin": 0.0}
    intersection = probability @ targets[:count].T
    union = (
        probability.sum(dim=-1)[:, None]
        + targets[:count].sum(dim=-1)[None]
        - intersection
    )
    similarity = intersection / union.clamp_min(1.0e-8)
    rows, columns = linear_sum_assignment(
        -similarity.detach().float().cpu().numpy()
    )
    selected = class_logits[0, rows].float()
    target = target_classes[0, columns].long()
    log_probability = selected.log_softmax(dim=-1)
    nll = -log_probability.gather(1, target[:, None]).mean()
    target_score = selected.gather(1, target[:, None])
    masked = selected.scatter(1, target[:, None], float("-inf"))
    margin = (target_score - masked.max(dim=-1, keepdim=True).values).mean()
    return {"nll": float(nll), "margin": float(margin)}


__all__ = [
    "matched_query_class_metrics",
    "multiclass_ece",
    "poisson_binomial_count_probability",
    "program_assignment_metrics",
    "soft_query_mask_iou",
]
