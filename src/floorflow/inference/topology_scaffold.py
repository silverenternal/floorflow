"""Guarded topology scaffold for local floorplan edits."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.fpeval_bridge import to_fpeval_plan
from floorflow.geometry.wall_graph_readout import raster_labels_to_wall_plan
from floorflow.inference.local_delta import compose_local_delta


def compose_topology_candidate(
    before: np.ndarray,
    after: np.ndarray,
    edit_region: np.ndarray,
) -> np.ndarray:
    """Compose a local topology candidate without target information."""
    return compose_local_delta(
        torch.from_numpy(before),
        torch.from_numpy(after),
        torch.from_numpy(edit_region),
    ).numpy()


def verify_topology(
    labels: np.ndarray,
    taxonomy: FloorplanTaxonomy,
) -> dict:
    """Return fpeval validity for a raster topology prediction."""
    plan = to_fpeval_plan(raster_labels_to_wall_plan(labels, taxonomy))
    from fpeval.verify import verify

    result = verify(plan)
    return result.as_dict()


def select_guarded_topology(
    before: np.ndarray,
    after: np.ndarray,
    edit_region: np.ndarray,
    taxonomy: FloorplanTaxonomy,
    *,
    verifier: Callable[[np.ndarray, FloorplanTaxonomy], dict] = verify_topology,
) -> tuple[np.ndarray, dict]:
    """Prefer the local scaffold, falling back when it is not watertight."""
    local = compose_topology_candidate(before, after, edit_region)
    local_validation = verifier(local, taxonomy)
    if bool(local_validation.get("watertight", False)):
        return local, {
            "source": "local_delta",
            "local_validity": local_validation,
            "fallback_validity": None,
        }
    fallback_validation = verifier(after, taxonomy)
    if bool(fallback_validation.get("watertight", False)):
        return after, {
            "source": "post_edit_fallback",
            "local_validity": local_validation,
            "fallback_validity": fallback_validation,
        }
    return local, {
        "source": "failed_both",
        "local_validity": local_validation,
        "fallback_validity": fallback_validation,
    }


__all__ = [
    "compose_topology_candidate",
    "select_guarded_topology",
    "verify_topology",
]
