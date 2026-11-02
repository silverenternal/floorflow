"""Thin adapters around maintained conditional-flow-matching implementations."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version

import torch
import torch.nn.functional as F

FLOW_MATCHERS = ("legacy", "torchcfm", "torchcfm_vp")
OT_MODES = ("none", "token")


@dataclass(frozen=True)
class FlowBatch:
    """A conditional probability-path sample used by a velocity regressor."""

    time: torch.Tensor
    location: torch.Tensor
    velocity: torch.Tensor


def flow_backend_version(backend: str = "torchcfm") -> str | None:
    """Return the installed package version for report provenance."""

    if backend == "legacy":
        return None
    try:
        return version("torchcfm")
    except PackageNotFoundError:
        return None


def sample_time(
    batch_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    sampling: str = "uniform",
    mean: float = 0.0,
    std: float = 1.0,
) -> torch.Tensor:
    """Sample the interpolation time with an explicit, testable distribution."""

    if sampling == "uniform":
        return torch.rand(batch_size, device=device, dtype=dtype)
    if sampling == "logit_normal":
        if std <= 0.0:
            raise ValueError("logit-normal std must be positive")
        normal = torch.randn(batch_size, device=device, dtype=dtype)
        return torch.sigmoid(float(mean) + float(std) * normal)
    raise ValueError(f"unsupported time sampling: {sampling}")


def sample_flow_batch(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    backend: str = "torchcfm",
    sigma: float = 0.0,
    time_sampling: str = "uniform",
    time_sampling_mean: float = 0.0,
    time_sampling_std: float = 1.0,
    minibatch_ot: str = "none",
    token_grid: int = 8,
) -> FlowBatch:
    """Sample ``(x_t, x_1-x_0)`` using the selected CFM implementation.

    The legacy implementation is retained for exact reproduction of historical
    runs. ``torchcfm`` delegates the same probability path to the maintained
    library; ``torchcfm_vp`` uses its trigonometric variance-preserving path.
    ``minibatch_ot="token"`` couples training samples through an exact
    minibatch assignment on pooled tokens (an OT-CFM-style pairing).  The cost
    matrix stays ``[B,B]``; it is never an all-pixel cost matrix.
    """

    if source.shape != target.shape:
        raise ValueError(f"source and target shapes differ: {source.shape} vs {target.shape}")
    if source.ndim < 2 or source.shape[0] != target.shape[0]:
        raise ValueError("source and target must have a shared non-empty batch dimension")
    if backend not in FLOW_MATCHERS:
        raise ValueError(f"unknown flow backend: {backend}")
    if minibatch_ot not in OT_MODES:
        raise ValueError(f"unknown minibatch OT mode: {minibatch_ot}")

    if minibatch_ot == "token" and source.shape[0] > 1:
        from floorflow.objectives.token_optimal_transport import (
            batch_token_assignment,
        )

        grid = max(1, int(token_grid))
        if source.ndim >= 3:
            source_tokens = F.adaptive_avg_pool2d(
                source.float(),
                (grid, grid),
            ).flatten(1)
            target_tokens = F.adaptive_avg_pool2d(
                target.float(),
                (grid, grid),
            ).flatten(1)
        else:
            source_tokens = source.float().flatten(1)
            target_tokens = target.float().flatten(1)
        order = batch_token_assignment(source_tokens, target_tokens)
        target = target[order]

    time = sample_time(
        source.shape[0],
        device=source.device,
        dtype=source.dtype,
        sampling=time_sampling,
        mean=time_sampling_mean,
        std=time_sampling_std,
    )
    if backend == "legacy":
        view = time[:, None, None, None]
        location = (1.0 - view) * source + view * target
        return FlowBatch(time=time, location=location, velocity=target - source)

    try:
        from torchcfm.conditional_flow_matching import (
            ConditionalFlowMatcher,
            VariancePreservingConditionalFlowMatcher,
        )
    except ImportError as error:
        raise ImportError(
            "TorchCFM is required for this matcher; install `planb[train]` "
            "or select --flow-matcher legacy"
        ) from error

    matcher_type = (
        VariancePreservingConditionalFlowMatcher
        if backend == "torchcfm_vp"
        else ConditionalFlowMatcher
    )
    matcher = matcher_type(sigma=max(0.0, float(sigma)))
    sampled_time, location, velocity = matcher.sample_location_and_conditional_flow(
        source,
        target,
        t=time,
    )
    return FlowBatch(time=sampled_time, location=location, velocity=velocity)
