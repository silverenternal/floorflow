"""Vector geometry, wall graphs, openings, and reconciliation."""

from __future__ import annotations

from importlib import import_module

__all__ = ["compose_known_exterior", "footprint_band", "interior_mask"]


def __getattr__(name: str):
    if name in __all__:
        return getattr(import_module(".core", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
