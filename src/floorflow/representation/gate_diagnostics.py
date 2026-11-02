"""Shared gate entropy and load-balance diagnostics."""

from __future__ import annotations

import torch


def stream_gate_diagnostics(gates: torch.Tensor) -> dict[str, torch.Tensor]:
    if gates.shape[-1] < 2:
        zero = gates.sum() * 0.0
        return {"gate_entropy": zero, "gate_load_balance": zero + 1.0}
    entropy = -(gates.clamp_min(1.0e-8).log() * gates).sum(dim=-1)
    normalized_entropy = entropy / torch.log(
        torch.tensor(float(gates.shape[-1]), device=gates.device, dtype=gates.dtype)
    )
    mean = gates.mean(dim=-1, keepdim=True)
    coefficient = gates.std(dim=-1, keepdim=True) / mean.clamp_min(1.0e-6)
    return {
        "gate_entropy": normalized_entropy.mean(),
        "gate_load_balance": (1.0 / (1.0 + coefficient)).mean(),
    }


__all__ = ["stream_gate_diagnostics"]
