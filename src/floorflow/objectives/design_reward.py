"""Structure-derived floorplan reward head and guidance."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from floorflow.evaluation.metrics import floorplan_metrics
from floorflow.data.taxonomy import FloorplanTaxonomy


class DesignRewardHead(nn.Module):
    def __init__(self, hidden_dim: int, metric_dim: int = 8, targets: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + metric_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, targets),
        )

    def forward(self, pooled_features: torch.Tensor, metrics: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([pooled_features, metrics], dim=-1))


def reward_guidance(prediction: torch.Tensor, reward: torch.Tensor, weight: float = 0.05) -> torch.Tensor:
    normalized = F.normalize(prediction.flatten(1), dim=1).view_as(prediction)
    return prediction + float(weight) * reward.mean(dim=1, keepdim=True).view(-1, 1, 1, 1) * normalized


def structural_reward_from_labels(
    prediction: torch.Tensor,
    target: torch.Tensor,
    footprint: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
) -> torch.Tensor:
    metrics = floorplan_metrics(prediction, target, footprint, taxonomy)
    return torch.tensor(
        [
            metrics["room_area_coverage"],
            metrics["opening_wall_alignment"],
            metrics["furniture_room_containment"],
            1.0 - metrics["room_area_distribution_mae"],
        ],
        device=prediction.device,
    )
