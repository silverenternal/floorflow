"""Local-delta rendering for interactive floorplan edits."""

from __future__ import annotations

import numpy as np
import torch


def compose_local_delta(
    before: torch.Tensor,
    after: torch.Tensor,
    edit_region: torch.Tensor,
) -> torch.Tensor:
    """Keep the cached pre-edit result outside a user-specified edit region."""

    if before.shape != after.shape:
        raise ValueError("before and after tensors must share a shape")
    if before.ndim == 2:
        if edit_region.ndim == 3 and edit_region.shape[0] == 1:
            edit_region = edit_region[0]
        if edit_region.shape != before.shape:
            raise ValueError("edit_region must match the spatial shape")
        return torch.where(edit_region.bool(), after, before)
    if edit_region.ndim == 2:
        edit_region = edit_region[None]
    if before.ndim == 3:
        expected = before.shape[-2:]
        if edit_region.shape not in {(1, *expected), before.shape}:
            raise ValueError("edit_region must match channel spatial shape")
        return torch.where(edit_region.bool(), after, before)
    if edit_region.ndim == before.ndim - 1:
        edit_region = edit_region[:, None]
    if edit_region.shape != before.shape[:1] + before.shape[-2:]:
        raise ValueError("edit_region must match the batch spatial shape")
    condition = (
        edit_region.bool()
        if before.ndim == 3
        else edit_region[:, None].bool()
    )
    return torch.where(condition, after, before)


def compose_typed_delta(
    before: np.ndarray,
    before_control: np.ndarray,
    after_control: np.ndarray,
    edit_region: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Repair only typed control changes while leaving all outside pixels exact."""

    if before.shape != before_control.shape or before.shape != after_control.shape:
        raise ValueError("before output and controls must share a spatial shape")
    if edit_region.shape != before.shape:
        raise ValueError("edit_region must match the control shape")
    changed = edit_region.astype(bool, copy=False) & (before_control != after_control)
    repaired_before = before.copy()
    repaired_before[changed] = before_control[changed]
    repaired_after = repaired_before.copy()
    repaired_after[changed] = after_control[changed]
    return repaired_before, repaired_after


__all__ = ["compose_local_delta", "compose_typed_delta"]
