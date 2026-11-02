"""Difficulty discriminator for adversarial dynamic mask curricula."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskDifficultyDiscriminator(nn.Module):
    """Predict how hard a masked coarse-control sample is to reconstruct."""

    def __init__(
        self,
        control_channels: int,
        hidden_channels: int = 24,
    ) -> None:
        super().__init__()
        hidden = int(hidden_channels)
        self.trunk = nn.Sequential(
            nn.Conv2d(int(control_channels) + 2, hidden, 3, padding=1),
            nn.GroupNorm(max(1, hidden // 8), hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden * 2, 3, stride=2, padding=1),
            nn.GroupNorm(max(1, hidden // 4), hidden * 2),
            nn.SiLU(),
            nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1),
            nn.GroupNorm(max(1, hidden // 4), hidden * 2),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.log_temperature = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        mask: torch.Tensor,
        difficulty: torch.Tensor,
        footprint: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if difficulty.ndim == 3:
            difficulty = difficulty[:, None]
        if footprint.ndim == 3:
            footprint = footprint[:, None]
        inputs = torch.cat(
            [mask.float(), difficulty.float(), footprint.float()],
            dim=1,
        )
        hidden = self.trunk(inputs)
        pooled = F.adaptive_avg_pool2d(hidden, (1, 1)).flatten(1)
        logits = self.head(pooled)
        temperature = self.log_temperature.exp().clamp(0.25, 4.0)
        calibrated_logits = logits / temperature
        return logits, {
            "predicted_hardness": calibrated_logits.sigmoid().squeeze(-1),
            "feature_norm": hidden.flatten(1).norm(dim=1),
            "temperature": temperature.detach(),
        }


def difficulty_rank_targets(difficulty: torch.Tensor) -> torch.Tensor:
    """Map per-sample reconstruction errors to within-batch percentile ranks."""

    values = difficulty.detach().float().flatten()
    if values.numel() <= 1:
        return torch.full_like(values, 0.5)
    order = values.argsort()
    ranks = torch.empty_like(values)
    ranks[order] = torch.arange(
        values.numel(),
        device=values.device,
        dtype=values.dtype,
    )
    return ranks / (values.numel() - 1)


def difficulty_rank_loss(
    logits: torch.Tensor,
    difficulty: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Train calibrated relative hardness with BCE and pairwise ranking."""

    logits = logits.flatten()
    targets = difficulty_rank_targets(difficulty).to(logits.device)
    if logits.numel() <= 1:
        zero = logits.sum() * 0.0
        return zero, {
            "difficulty_bce": zero.detach(),
            "difficulty_pair": zero.detach(),
            "difficulty_spearman": zero.detach(),
        }
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    difference = difficulty.detach().float().flatten()[:, None] - difficulty.detach().float().flatten()[None, :]
    pair_valid = difference.abs() > 1.0e-8
    pair_sign = difference.sign()
    pair_logits = logits[:, None] - logits[None, :]
    pair = F.softplus(-pair_sign * pair_logits)[pair_valid].mean()
    predicted = logits.detach().float().cpu().numpy()
    observed = difficulty.detach().float().flatten().cpu().numpy()
    if np.std(predicted) < 1.0e-8 or np.std(observed) < 1.0e-8:
        spearman = 0.0
    else:
        spearman = float(np.corrcoef(predicted, observed)[0, 1])
    total = bce + 0.5 * pair
    return total, {
        "difficulty_bce": bce.detach(),
        "difficulty_pair": pair.detach(),
        "difficulty_spearman": torch.tensor(
            spearman,
            device=logits.device,
        ),
    }


__all__ = [
    "MaskDifficultyDiscriminator",
    "difficulty_rank_loss",
    "difficulty_rank_targets",
]
