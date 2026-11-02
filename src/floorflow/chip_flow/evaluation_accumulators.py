"""Shared accumulator schemas for chip refiner evaluation."""

from __future__ import annotations

PHYSICS_KEYS = (
    "overlap",
    "mib",
    "boundary",
    "wl",
    "area",
    "b2b",
    "p2b",
    "total",
    "group",
)


def physics_accumulator() -> dict[str, float]:
    return {name: 0.0 for name in PHYSICS_KEYS}


__all__ = ["PHYSICS_KEYS", "physics_accumulator"]
