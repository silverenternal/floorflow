"""Target-free router for single-trajectory legalizer portfolio profiles."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

PROFILE_NAMES = ("baseline", "projector32", "expanded")


@dataclass(frozen=True)
class PortfolioProfile:
    name: str
    projector_iterations: int
    relation: str


PROFILES = (
    PortfolioProfile("baseline", 8, "restored"),
    PortfolioProfile("projector32", 32, "restored"),
    PortfolioProfile("expanded", 8, "expanded"),
)


def router_features(
    constraints: torch.Tensor,
    b2b: torch.Tensor,
    p2b: torch.Tensor,
    n_b2b: torch.Tensor,
    n_p2b: torch.Tensor,
    areas: torch.Tensor | None = None,
    pins: torch.Tensor | None = None,
    n_pins: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build normalized complexity features available before generation."""

    count = constraints.shape[1]
    b2b_density = n_b2b.float() / count
    p2b_density = n_p2b.float() / count
    anchor_fraction = constraints[..., :2].ne(0).any(dim=-1).float().mean(dim=1)
    cluster_fraction = constraints[..., 3].ne(0).float().mean(dim=1)
    b2b_mean, b2b_std = _edge_stats(b2b, n_b2b)
    p2b_mean, p2b_std = _edge_stats(p2b, n_p2b)
    if areas is None:
        area_cv = b2b_density.new_zeros(batch_shape := b2b_density.shape)
    else:
        batch_shape = areas.shape
        area_cv = (
            areas.float().std(dim=1, correction=0)
            / areas.float().mean(dim=1).clamp_min(1.0e-6)
        )
    pin_spread = b2b_density.new_zeros(batch_shape)
    if pins is not None and pins.numel():
        pin_rows = torch.arange(pins.shape[1], device=pins.device)[None]
        valid = (
            pin_rows < n_pins.to(pins.device).long()[:, None]
            if n_pins is not None
            else torch.ones(
                pins.shape[0],
                pins.shape[1],
                device=pins.device,
                dtype=torch.bool,
            )
        )
        count = valid.sum(dim=1).clamp_min(1)
        mean = (pins.float() * valid[..., None]).sum(dim=1) / count[:, None]
        variance = (
            (pins.float() - mean[:, None]).square() * valid[..., None]
        ).sum(dim=(1, 2)) / count
        pin_spread = variance.sqrt().clamp(0.0, 4.0) / 4.0
    return torch.stack(
        [
            torch.log1p(torch.as_tensor(count, device=constraints.device))
            .expand_as(b2b_density),
            b2b_density,
            p2b_density,
            anchor_fraction + cluster_fraction,
            area_cv.clamp(0.0, 4.0) / 4.0,
            b2b_mean.clamp(0.0, 16.0) / 16.0,
            b2b_std.clamp(0.0, 16.0) / 16.0,
            p2b_mean.clamp(0.0, 16.0) / 16.0,
            p2b_std.clamp(0.0, 16.0) / 16.0,
            pin_spread,
        ],
        dim=-1,
    )


def _edge_stats(
    edges: torch.Tensor,
    lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = edges.shape[0]
    if edges.shape[1] == 0:
        zeros = edges.new_zeros(batch)
        return zeros, zeros
    rows = torch.arange(edges.shape[1], device=edges.device)[None]
    valid = rows < lengths.to(edges.device).long()[:, None]
    weight = edges[..., 2].float().clamp_min(0.0)
    count = valid.sum(dim=1).clamp_min(1)
    mean = (weight * valid).sum(dim=1) / count
    variance = (
        (weight - mean[:, None]).square() * valid
    ).sum(dim=1) / count
    return mean, variance.sqrt()


class PortfolioRouter(nn.Module):
    """Small router that chooses one portfolio profile before generation."""

    def __init__(self, hidden: int = 32, feature_dim: int = 10) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.network = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, len(PROFILES)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != self.feature_dim:
            raise ValueError("portfolio router feature width mismatch")
        return self.network(features.float())

    def choose(self, features: torch.Tensor) -> tuple[PortfolioProfile, ...]:
        indices = self(features).argmax(dim=-1).detach().cpu().tolist()
        return tuple(PROFILES[int(index)] for index in indices)


def validate_router_accuracy(
    accuracy: float,
    minimum: float,
    *,
    allow_low_accuracy: bool = False,
) -> None:
    """Fail closed unless the router passes its held-out promotion gate."""

    if accuracy < minimum and not allow_low_accuracy:
        raise ValueError(
            f"router validation accuracy {accuracy:.3f} is below {minimum:.3f}; "
            "use the balanced portfolio or opt in to a diagnostic run"
        )


__all__ = [
    "PROFILES",
    "PROFILE_NAMES",
    "PortfolioProfile",
    "PortfolioRouter",
    "router_features",
    "validate_router_accuracy",
]
