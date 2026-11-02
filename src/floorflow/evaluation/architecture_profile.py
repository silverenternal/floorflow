"""Resource profiling for paper-facing model component audits."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import torch

from floorflow.model.budget import count_parameters, profile_forward_macs


@dataclass(frozen=True)
class ResourceProfile:
    parameters: int
    macs: int
    latency_ms_mean: float
    latency_ms_p50: float
    peak_memory_bytes: int | None


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def profile_callable(
    module: torch.nn.Module,
    forward: Callable[[], object],
    *,
    device: torch.device,
    warmup: int = 5,
    repeats: int = 20,
) -> ResourceProfile:
    """Measure deterministic parameters, MACs, latency, and peak GPU memory."""

    parameter_count = count_parameters(module)
    macs = profile_forward_macs(module, forward).macs
    is_cuda = device.type == "cuda"
    if is_cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        baseline = torch.cuda.memory_allocated(device)
    with torch.inference_mode():
        for _ in range(int(warmup)):
            forward()
        if is_cuda:
            torch.cuda.synchronize(device)
        samples = []
        for _ in range(int(repeats)):
            started = time.perf_counter()
            forward()
            if is_cuda:
                torch.cuda.synchronize(device)
            samples.append((time.perf_counter() - started) * 1000.0)
    peak = (
        int(torch.cuda.max_memory_allocated(device) - baseline)
        if is_cuda
        else None
    )
    return ResourceProfile(
        parameters=parameter_count,
        macs=macs,
        latency_ms_mean=sum(samples) / len(samples),
        latency_ms_p50=_percentile(samples, 0.5),
        peak_memory_bytes=peak,
    )


def resource_dict(profile: ResourceProfile) -> dict[str, int | float | None]:
    return {
        "parameters": profile.parameters,
        "macs": profile.macs,
        "gmacs": profile.macs / 1.0e9,
        "latency_ms_mean": profile.latency_ms_mean,
        "latency_ms_p50": profile.latency_ms_p50,
        "peak_memory_bytes": profile.peak_memory_bytes,
    }


__all__ = ["ResourceProfile", "profile_callable", "resource_dict"]
