"""Reproducible wrapper around Trinity's existing scale_pack legalizer."""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np
from trinity.floorplan.legalize import pack_lp
from trinity.floorplan.legalize.scale_pack import scale_pack_legalize
from trinity.floorplan.types import Placement

EVALUATION_TIME_LIMIT_MS = 600_000


@contextmanager
def _single_threaded_glop():
    """Force deterministic GLOP scheduling for the duration of one solve."""

    original_init = pack_lp._PackingLP.__init__

    def deterministic_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if self.ok:
            self.solver.SetNumThreads(1)

    pack_lp._PackingLP.__init__ = deterministic_init
    try:
        yield
    finally:
        pack_lp._PackingLP.__init__ = original_init


def scale_pack_legalize_reproducible(
    placement: Placement,
    time_limit_ms: int = EVALUATION_TIME_LIMIT_MS,
    **kwargs,
) -> Placement:
    """Run scale_pack without the short LP timeout that makes ties unstable.

    The pinned Trinity implementation uses a two-second time limit. A feasible
    but non-optimal LP solution can change between runs when that limit fires,
    which changes the official score even for identical model predictions.
    The solver remains the same; only the evaluation timeout is removed.
    """

    if time_limit_ms <= 0:
        raise ValueError("time_limit_ms must be positive")
    previous = pack_lp.TIME_LIMIT_MS
    pack_lp.TIME_LIMIT_MS = int(time_limit_ms)
    try:
        with _single_threaded_glop():
            return scale_pack_legalize(placement, **kwargs)
    finally:
        pack_lp.TIME_LIMIT_MS = previous


def translate_into_core(
    boxes: np.ndarray,
    core: tuple[float, float, float, float],
    *,
    tolerance: float = 1.0e-6,
) -> np.ndarray | None:
    """Rigidly translate a packed layout into the core when its bbox fits."""

    out = np.asarray(boxes, dtype=np.float64).copy()
    x0, y0, x1, y1 = core
    minimum_x = float(out[:, 0].min())
    minimum_y = float(out[:, 1].min())
    maximum_x = float((out[:, 0] + out[:, 2]).max())
    maximum_y = float((out[:, 1] + out[:, 3]).max())
    if maximum_x - minimum_x > x1 - x0 + tolerance:
        return None
    if maximum_y - minimum_y > y1 - y0 + tolerance:
        return None
    dx = (
        x0 - minimum_x
        if minimum_x < x0 - tolerance
        else x1 - maximum_x
        if maximum_x > x1 + tolerance
        else 0.0
    )
    dy = (
        y0 - minimum_y
        if minimum_y < y0 - tolerance
        else y1 - maximum_y
        if maximum_y > y1 + tolerance
        else 0.0
    )
    out[:, 0] += dx
    out[:, 1] += dy
    if out[:, 0].min() < x0 - tolerance or out[:, 1].min() < y0 - tolerance:
        return None
    if (out[:, 0] + out[:, 2]).max() > x1 + tolerance:
        return None
    if (out[:, 1] + out[:, 3]).max() > y1 + tolerance:
        return None
    return out


__all__ = [
    "EVALUATION_TIME_LIMIT_MS",
    "scale_pack_legalize_reproducible",
    "translate_into_core",
]
