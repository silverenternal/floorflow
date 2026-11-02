"""Target-free fusion of two contour-conditioned distribution outputs."""

from __future__ import annotations

from typing import Any

import torch


def _blend_tree(
    primary: Any,
    secondary: Any,
    weight: float,
) -> Any:
    if (
        isinstance(primary, torch.Tensor)
        and isinstance(secondary, torch.Tensor)
        and primary.shape == secondary.shape
        and primary.is_floating_point()
        and secondary.is_floating_point()
    ):
        return (1.0 - weight) * primary + weight * secondary.to(
            device=primary.device,
            dtype=primary.dtype,
        )
    if isinstance(primary, dict) and isinstance(secondary, dict):
        return {
            key: _blend_tree(value, secondary[key], weight)
            if key in secondary
            else value
            for key, value in primary.items()
        }
    return primary


def blend_distribution_outputs(
    primary: dict[str, Any],
    secondary: dict[str, Any],
    weight: float,
) -> dict[str, Any]:
    """Blend compatible tensor states while preserving primary-only metadata."""

    if not 0.0 <= float(weight) <= 1.0:
        raise ValueError("ensemble weight must be in [0, 1]")
    return _blend_tree(primary, secondary, float(weight))


__all__ = ["blend_distribution_outputs"]
