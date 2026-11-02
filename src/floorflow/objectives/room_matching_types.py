"""Shared room-query assignment contracts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RoomQueryMatch:
    """One-to-one assignment between predicted and target room indices."""

    predicted_indices: np.ndarray
    target_indices: np.ndarray


__all__ = ["RoomQueryMatch"]

