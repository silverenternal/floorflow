"""Target-free candidate features and legalization-risk reranking."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

FEATURE_NAMES = (
    "v_rel_rank",
    "displacement_rank",
    "boundary_rank",
    "grouping_rank",
    "b2b_hpwl_rank",
    "p2b_hpwl_rank",
    "compactness_rank",
    "aspect_rank",
    "rung_rank",
    "solves_rank",
)


def target_free_candidate_features(entries: Sequence[dict]) -> torch.Tensor:
    """Build within-case rank features without target positions or official cost."""

    if not entries:
        raise ValueError("candidate feature extraction requires at least one entry")
    values = {
        "v_rel": [float(entry["stages"]["legalized"]["v_rel"]) for entry in entries],
        "displacement": [
            float(entry["displacement"]["center_l2"]) for entry in entries
        ],
        "boundary": [
            float(entry["stages"]["legalized"]["boundary"]) for entry in entries
        ],
        "grouping": [
            float(entry["stages"]["legalized"]["grouping"]) for entry in entries
        ],
        "b2b_hpwl": [
            float(entry["target_free_geometry"]["b2b_hpwl"]) for entry in entries
        ],
        "p2b_hpwl": [
            float(entry["target_free_geometry"]["p2b_hpwl"]) for entry in entries
        ],
        "compactness": [
            float(entry["target_free_geometry"]["outline_compactness"])
            for entry in entries
        ],
        "aspect": [
            abs(float(entry["target_free_geometry"]["outline_aspect"]))
            for entry in entries
        ],
        "rung": [float(entry["legalizer"]["rung"]) for entry in entries],
        "solves": [float(entry["legalizer"]["solves"]) for entry in entries],
    }
    columns = [
        _rank_feature(values["v_rel"]),
        _rank_feature(values["displacement"]),
        _rank_feature(values["boundary"]),
        _rank_feature(values["grouping"]),
        _rank_feature(values["b2b_hpwl"]),
        _rank_feature(values["p2b_hpwl"]),
        _rank_feature(values["compactness"], higher_is_better=True),
        _rank_feature(values["aspect"]),
        _rank_feature(values["rung"]),
        _rank_feature(values["solves"]),
    ]
    return torch.stack(columns, dim=-1).float()


def _rank_feature(values: list[float], *, higher_is_better: bool = False) -> torch.Tensor:
    tensor = torch.tensor(values, dtype=torch.float64)
    oriented = -tensor if higher_is_better else tensor
    order = torch.argsort(oriented, stable=True)
    sorted_values = oriented[order]
    sorted_ranks = torch.empty_like(oriented)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        sorted_ranks[start:end] = (start + end - 1) / 2.0
        start = end
    ranks = torch.empty_like(oriented)
    ranks[order] = sorted_ranks
    denominator = max(len(values) - 1, 1)
    return (ranks / denominator).to(torch.float32)


class CandidateLegalizabilityReranker(nn.Module):
    """Small permutation-equivariant listwise candidate scorer."""

    def __init__(self, hidden: int = 16, feature_dim: int = len(FEATURE_NAMES)) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.network = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != self.feature_dim:
            raise ValueError("candidate feature width mismatch")
        return self.network(features.float()).squeeze(-1)


def legalized_cost_targets(entries: Sequence[dict]) -> torch.Tensor:
    return torch.tensor(
        [float(entry["stages"]["legalized"]["cost"]) for entry in entries],
        dtype=torch.float32,
    )


def listwise_ranking_loss(
    scores: torch.Tensor,
    costs: torch.Tensor,
    *,
    temperature: float = 0.05,
) -> torch.Tensor:
    if scores.shape != costs.shape:
        raise ValueError("candidate scores and costs must have matching shapes")
    target = torch.softmax(-costs.detach() / float(temperature), dim=-1)
    return -(target * torch.log_softmax(scores, dim=-1)).sum()


def select_candidate(scores: torch.Tensor) -> int:
    if scores.ndim != 1 or not scores.numel():
        raise ValueError("candidate scores must be a non-empty vector")
    return int(torch.argmin(scores).item())


__all__ = [
    "FEATURE_NAMES",
    "CandidateLegalizabilityReranker",
    "legalized_cost_targets",
    "listwise_ranking_loss",
    "select_candidate",
    "target_free_candidate_features",
]
