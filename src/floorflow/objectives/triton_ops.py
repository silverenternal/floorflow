"""Optional Triton kernels with deterministic Torch fallbacks."""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised only without Triton
    triton = None
    tl = None


def triton_available() -> bool:
    return triton is not None


def _kernel_enabled(enabled: bool | None, tensor: torch.Tensor) -> bool:
    if triton is None or tensor.device.type != "cuda":
        return False
    if enabled is not None:
        return bool(enabled)
    return os.environ.get("FLOORFLOW_TRITON_KERNELS", "0") == "1"


if triton is not None:

    @triton.jit
    def _vq_indices_kernel(
        values,
        codebook,
        indices,
        D: tl.constexpr,
        K: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_D)
        mask = offsets < D
        value = tl.load(values + row * D + offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        best_distance = float("inf")
        best_index = 0
        for code_index in range(K):
            code = tl.load(
                codebook + code_index * D + offsets,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            distance = tl.sum((value - code) * (value - code), axis=0)
            take = distance < best_distance
            best_distance = tl.where(take, distance, best_distance)
            best_index = tl.where(take, code_index, best_index)
        tl.store(indices + row, best_index)

    @triton.jit
    def _soft_incidence_kernel(
        points,
        endpoints,
        output,
        J: tl.constexpr,
        E: tl.constexpr,
        D: tl.constexpr,
        THRESHOLD: tl.constexpr,
        SHARPNESS: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        batch = tl.program_id(0)
        junction = tl.program_id(1)
        endpoint = tl.program_id(2)
        offsets = tl.arange(0, BLOCK_D)
        mask = offsets < D
        point = tl.load(
            points + (batch * J + junction) * D + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        endpoint_value = tl.load(
            endpoints + (batch * E + endpoint) * D + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        difference = point - endpoint_value
        distance = tl.sqrt(tl.sum(difference * difference, axis=0))
        probability = 1.0 / (
            1.0 + tl.exp(-(THRESHOLD - distance) * SHARPNESS)
        )
        tl.store(output + (batch * J + junction) * E + endpoint, probability)


def triton_vq_indices(
    values: torch.Tensor,
    codebook: torch.Tensor,
    *,
    enabled: bool | None = None,
) -> torch.Tensor:
    flat = values.reshape(-1, values.shape[-1])
    if not _kernel_enabled(enabled, flat):
        return torch.cdist(flat, codebook).argmin(dim=-1)
    if not flat.is_contiguous():
        flat = flat.contiguous()
    codebook = codebook.contiguous()
    indices = torch.empty(
        flat.shape[0],
        dtype=torch.long,
        device=flat.device,
    )
    block = triton.next_power_of_2(int(flat.shape[1]))
    _vq_indices_kernel[(flat.shape[0],)](
        flat,
        codebook,
        indices,
        D=int(flat.shape[1]),
        K=int(codebook.shape[0]),
        BLOCK_D=block,
        num_warps=4,
    )
    return indices


def triton_vq_quantize(
    values: torch.Tensor,
    codebook: torch.Tensor,
    *,
    enabled: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flat = values.reshape(-1, values.shape[-1])
    indices = triton_vq_indices(flat, codebook, enabled=enabled)
    quantized = F.embedding(indices, codebook).reshape_as(values)
    straight_through = values + (quantized - values).detach()
    commitment = F.mse_loss(values, quantized.detach())
    codebook_loss = F.mse_loss(quantized, values.detach())
    return (
        straight_through,
        indices.reshape(values.shape[:-1]),
        commitment + codebook_loss,
    )


def endpoint_incidence(
    points: torch.Tensor,
    endpoints: torch.Tensor,
    *,
    threshold: float = 0.02,
    sharpness: float = 100.0,
    enabled: bool | None = None,
) -> torch.Tensor:
    if points.ndim != 3 or endpoints.ndim != 3:
        raise ValueError("points and endpoints must be [B,N,D]")
    if points.shape[0] != endpoints.shape[0]:
        raise ValueError("point and endpoint batches must match")
    difference = points[:, :, None, :] - endpoints[:, None, :, :]
    distance = difference.square().sum(dim=-1).sqrt()
    fallback = torch.sigmoid((float(threshold) - distance) * float(sharpness))
    if not _kernel_enabled(enabled, points) or points.requires_grad:
        return fallback
    output = torch.empty(
        (points.shape[0], points.shape[1], endpoints.shape[1]),
        dtype=torch.float32,
        device=points.device,
    )
    block = triton.next_power_of_2(int(points.shape[-1]))
    _soft_incidence_kernel[
        (points.shape[0], points.shape[1], endpoints.shape[1])
    ](
        points.contiguous(),
        endpoints.contiguous(),
        output,
        J=int(points.shape[1]),
        E=int(endpoints.shape[1]),
        D=int(points.shape[2]),
        THRESHOLD=float(threshold),
        SHARPNESS=float(sharpness),
        BLOCK_D=block,
        num_warps=4,
    )
    return output


def triton_status() -> dict[str, bool | str]:
    return {
        "available": triton_available(),
        "requested": os.environ.get("FLOORFLOW_TRITON_KERNELS", "0") == "1",
        "version": getattr(triton, "__version__", "unavailable"),
    }


__all__ = [
    "endpoint_incidence",
    "triton_available",
    "triton_status",
    "triton_vq_indices",
    "triton_vq_quantize",
]
