"""Execution controls for deterministic chip-model evaluation."""

from __future__ import annotations

import os

import torch


def stable_batch_seed(base_seed: int, indices) -> int:
    """Return one repeatable seed per evaluation batch without cross-batch reuse."""

    values = [int(value) for value in indices]
    if not values:
        raise ValueError("batch seed requires at least one case index")
    return (int(base_seed) * 1_000_003 + values[0] * 104_729) % (2**63 - 1)


def enable_deterministic_execution() -> None:
    """Request deterministic CUDA kernels and disable benchmark-dependent choices."""

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


__all__ = ["enable_deterministic_execution", "stable_batch_seed"]
