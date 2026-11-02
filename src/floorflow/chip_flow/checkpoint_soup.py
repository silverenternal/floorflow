"""Weighted checkpoint averaging for same-architecture chip-flow models."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def weighted_average_state_dicts(
    states: Sequence[dict[str, torch.Tensor]],
    weights: Sequence[float] | None = None,
) -> dict[str, torch.Tensor]:
    """Average floating tensors and preserve identical non-floating buffers."""

    if not states:
        raise ValueError("checkpoint soup requires at least one state dictionary")
    if weights is None:
        weights = [1.0] * len(states)
    if len(weights) != len(states):
        raise ValueError("weights must match the number of state dictionaries")
    total = float(sum(weights))
    if total <= 0.0:
        raise ValueError("checkpoint soup weights must sum to a positive value")
    key_set = set(states[0])
    for state in states[1:]:
        if set(state) != key_set:
            missing = sorted(key_set.symmetric_difference(state))
            raise ValueError(f"checkpoint state keys differ: {missing[:8]}")
    output = {}
    for key in sorted(key_set):
        values = [state[key] for state in states]
        shape = values[0].shape
        if any(value.shape != shape for value in values[1:]):
            raise ValueError(f"checkpoint tensor shape mismatch for {key}")
        if values[0].is_floating_point():
            averaged = sum(
                weight * value.float()
                for weight, value in zip(weights, values, strict=True)
            ) / total
            output[key] = averaged.to(dtype=values[0].dtype).detach().clone()
        else:
            output[key] = values[0].detach().clone()
    return output


__all__ = ["weighted_average_state_dicts"]
