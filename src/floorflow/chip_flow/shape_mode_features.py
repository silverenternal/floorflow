"""Target-free geometric descriptors for shape-mode selection."""

from __future__ import annotations

import torch

from floorflow.chip_flow.batched import batch_physics_terms
from floorflow.chip_flow.model import decode_boxes_batch

SHAPE_MODE_FEATURE_NAMES = (
    "b2b",
    "p2b",
    "overlap",
    "area",
    "wl",
    "mib",
    "boundary",
    "rho_delta",
)


def shape_mode_geometry_features(
    mode_latents: torch.Tensor,
    baseline: torch.Tensor,
    batch: dict,
    *,
    include_baseline: bool = False,
) -> torch.Tensor:
    """Describe each shape candidate using inference-visible geometry only."""

    if mode_latents.ndim != 4 or mode_latents.shape[-1] != 3:
        raise ValueError("mode_latents must be [B,M,N,3]")
    if baseline.shape != mode_latents.shape[:1] + mode_latents.shape[2:]:
        raise ValueError("baseline must be [B,N,3]")
    batch_size, _modes, count, _ = mode_latents.shape
    candidates = (
        torch.cat([baseline[:, None], mode_latents], dim=1)
        if include_baseline
        else mode_latents
    )
    candidate_count = candidates.shape[1]
    flat_modes = candidates.reshape(batch_size * candidate_count, count, 3)
    flat_areas = batch["areas"].repeat_interleave(candidate_count, dim=0)
    flat_boxes = decode_boxes_batch(flat_modes, flat_areas)
    terms = batch_physics_terms(
        flat_boxes,
        flat_areas,
        batch["constraints"].repeat_interleave(candidate_count, dim=0),
        batch["b2b"].repeat_interleave(candidate_count, dim=0),
        batch["p2b"].repeat_interleave(candidate_count, dim=0),
        batch["pins"].repeat_interleave(candidate_count, dim=0),
        batch["n_b2b"].repeat_interleave(candidate_count, dim=0),
        batch["n_p2b"].repeat_interleave(candidate_count, dim=0),
        batch["n_pins"].repeat_interleave(candidate_count, dim=0),
    )
    descriptors = torch.stack(
        [terms[name] for name in SHAPE_MODE_FEATURE_NAMES[:-1]],
        dim=-1,
    ).reshape(batch_size, candidate_count, -1)
    rho_delta = (
        candidates[..., 2] - baseline[:, None, :, 2]
    ).abs().mean(dim=-1, keepdim=True)
    return torch.cat([descriptors, rho_delta], dim=-1)


def select_pareto_safe_shape_mode(
    candidate_features: torch.Tensor,
    *,
    tolerance: float = 0.01,
    minimum_improvement: float = 0.01,
) -> torch.Tensor:
    """Return a conservative candidate index with the baseline at index zero."""

    if candidate_features.ndim != 3 or candidate_features.shape[1] < 2:
        raise ValueError("candidate_features must be [B,C>=2,D]")
    if (
        candidate_features.shape[-1] != len(SHAPE_MODE_FEATURE_NAMES)
        or tolerance < 0.0
        or minimum_improvement < 0.0
    ):
        raise ValueError("invalid candidate feature selection contract")
    baseline = candidate_features[:, :1, :7]
    relative = (
        candidate_features[..., :7] - baseline
    ) / baseline.abs().clamp_min(1.0e-6)
    non_regressing = relative.le(float(tolerance)).all(dim=-1)
    improving = relative.lt(-float(minimum_improvement)).any(dim=-1)
    eligible = non_regressing & improving
    weights = candidate_features.new_tensor(
        [0.20, 0.20, 0.20, 0.20, 0.10, 0.03, 0.07]
    )
    score = (relative * weights).sum(dim=-1)
    score = score.masked_fill(~eligible, float("inf"))
    selected = score.argmin(dim=-1)
    no_candidate = ~eligible.any(dim=-1)
    return torch.where(
        no_candidate,
        torch.zeros_like(selected),
        selected,
    )


__all__ = [
    "SHAPE_MODE_FEATURE_NAMES",
    "select_pareto_safe_shape_mode",
    "shape_mode_geometry_features",
]
