"""Boundary-conditioned floorplan flow matching."""

from floorflow._compat import install_legacy_aliases

install_legacy_aliases()

from floorflow.data.classes import (
    CLASS_COLORS,
    CLASS_NAMES,
    FURNITURE_CLASS_IDS,
    NUM_CLASSES,
    NUM_TARGET_CHANNELS,
    ROOM_CLASS_IDS,
)

__all__ = [
    "CLASS_COLORS",
    "CLASS_NAMES",
    "FURNITURE_CLASS_IDS",
    "NUM_CLASSES",
    "NUM_TARGET_CHANNELS",
    "ROOM_CLASS_IDS",
]
