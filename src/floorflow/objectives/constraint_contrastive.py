"""Multi-positive contrastive and constraint-ranking training objectives."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def multi_positive_infonce(
    embeddings: torch.Tensor,
    positive_mask: torch.Tensor,
    *,
    temperature: float = 0.07,
) -> torch.Tensor:
    """InfoNCE with any number of same-group positives per anchor."""

    if embeddings.ndim != 2 or positive_mask.shape != (
        embeddings.shape[0],
        embeddings.shape[0],
    ):
        raise ValueError("embeddings and positive_mask must share [B,B] groups")
    logits = embeddings @ embeddings.T / max(float(temperature), 1.0e-6)
    mask = positive_mask.bool().clone()
    mask.fill_diagonal_(False)
    has_positive = mask.any(dim=1)
    if not bool(has_positive.any()):
        return embeddings.sum() * 0.0
    log_probability = logits.log_softmax(dim=1)
    per_anchor = -(
        log_probability.masked_fill(~mask, 0.0).sum(dim=1)
        / mask.sum(dim=1).clamp_min(1)
    )
    return per_anchor[has_positive].mean()


def multi_label_violation_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    positive = targets.sum(dim=0)
    negative = targets.shape[0] - positive
    pos_weight = (negative / positive.clamp_min(1.0)).clamp(1.0, 50.0)
    return F.binary_cross_entropy_with_logits(
        logits,
        targets.float(),
        pos_weight=pos_weight,
    )


def energy_ranking_loss(
    energy: torch.Tensor,
    legal: torch.Tensor,
    *,
    margin: float = 1.0,
) -> torch.Tensor:
    """Penalize legal states with higher energy than illegal states."""

    legal_energy = energy[legal.bool()]
    illegal_energy = energy[~legal.bool()]
    if legal_energy.numel() == 0 or illegal_energy.numel() == 0:
        return energy.sum() * 0.0
    pairwise = (
        margin
        + legal_energy[:, None]
        - illegal_energy[None, :]
    ).relu()
    return pairwise.mean()


__all__ = [
    "energy_ranking_loss",
    "multi_label_violation_loss",
    "multi_positive_infonce",
]
