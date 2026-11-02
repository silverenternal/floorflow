"""Regularization for multi-relation gate usage."""

from __future__ import annotations

import torch


def relation_gate_regularization(module) -> torch.Tensor:
    """Encourage nonzero, non-collapsed use of relation families."""

    gates = getattr(module, "relation_gate", None)
    if gates is None:
        return torch.zeros(())
    magnitude = gates.float().square()
    probability = torch.softmax(magnitude, dim=0)
    entropy = -(probability * probability.clamp_min(1.0e-8).log()).sum()
    entropy = entropy / torch.log(gates.new_tensor(float(len(gates))))
    return 0.1 * (1.0 - entropy) + (0.05 - magnitude.mean()).relu()


__all__ = ["relation_gate_regularization"]
