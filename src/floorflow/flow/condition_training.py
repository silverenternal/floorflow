"""Conditional flow path construction for structured floorplan training."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from floorflow.flow.cfm_backend import sample_flow_batch
from floorflow.flow.core import sample_prior, target_to_continuous
from floorflow.recursion.inward_flow import contour_inward_weight_map


@dataclass(frozen=True)
class ConditionFlowSample:
    source: torch.Tensor
    target: torch.Tensor
    location: torch.Tensor
    time: torch.Tensor
    velocity: torch.Tensor
    weight_map: torch.Tensor | None = None


def build_condition_flow_sample(
    labels: torch.Tensor,
    footprint: torch.Tensor,
    *,
    channels: int,
    matcher: str = "torchcfm",
    time_sampling: str = "uniform",
    noise_scale: float = 0.75,
    sigma: float = 0.0,
    inward_progress: float | None = None,
    inward_band_width: float = 0.22,
) -> ConditionFlowSample:
    source = sample_prior(
        footprint,
        channels,
        noise_scale=noise_scale,
    )
    target = target_to_continuous(labels, channels)
    flow = sample_flow_batch(
        source,
        target,
        backend=matcher,
        sigma=sigma,
        time_sampling=time_sampling,
    )
    weight_map = (
        contour_inward_weight_map(
            footprint,
            inward_progress,
            band_width=inward_band_width,
        )
        if inward_progress is not None
        else None
    )
    return ConditionFlowSample(
        source=source,
        target=target,
        location=flow.location,
        time=flow.time,
        velocity=flow.velocity,
        weight_map=weight_map,
    )


def condition_velocity_loss(
    predicted: torch.Tensor,
    expected: torch.Tensor,
    footprint: torch.Tensor,
    weight_map: torch.Tensor | None = None,
) -> torch.Tensor:
    mask = footprint[:, None] if footprint.ndim == 3 else footprint
    weights = mask
    if weight_map is not None:
        weights = weight_map
        if weights.ndim == 2:
            weights = weights[None]
        if weights.ndim == 4:
            weights = weights[:, 0]
        weights = weights[:, None].to(mask.device, mask.dtype)
        weights = weights * mask
    weighted = (predicted - expected).pow(2) * weights
    return weighted.sum() / (
        weights.sum().clamp_min(1.0) * max(1, predicted.shape[1])
    )
