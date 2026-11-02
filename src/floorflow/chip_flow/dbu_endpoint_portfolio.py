"""Target-free portfolio over DBU legalizer relations and contact references."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from trinity.floorplan.types import FloorplanInstance

from floorflow.chip_flow.dbu_projection import (
    DbuProjectionResult,
    project_quantized_layout,
)


@dataclass(frozen=True)
class DbuEndpointCandidate:
    name: str
    result: DbuProjectionResult
    displacement_p50: float
    displacement_p95: float
    max_displacement: float


@dataclass(frozen=True)
class DbuEndpointPortfolioResult:
    selected: DbuEndpointCandidate
    candidates: tuple[DbuEndpointCandidate, ...]
    tried_routes: tuple[str, ...]


def displacement_summary(
    before: np.ndarray,
    after: np.ndarray,
) -> tuple[float, float, float]:
    delta = np.asarray(after)[:, :2] - np.asarray(before)[:, :2]
    translation = np.median(delta, axis=0)
    distance = np.linalg.norm(delta - translation, axis=1)
    return (
        float(np.quantile(distance, 0.50)),
        float(np.quantile(distance, 0.95)),
        float(distance.max(initial=0.0)),
    )


def _candidate(
    name: str,
    boxes: np.ndarray,
    instance: FloorplanInstance,
    *,
    relation: str,
    contact_reference: np.ndarray | None,
    resolution: float,
    legalization_time_limit_ms: int,
    projection_time_limit_ms: int,
) -> DbuEndpointCandidate:
    result = project_quantized_layout(
        boxes,
        instance,
        resolution=resolution,
        legalization_time_limit_ms=legalization_time_limit_ms,
        projection_time_limit_ms=projection_time_limit_ms,
        contact_reference=contact_reference,
        legalizer_relation=relation,
    )
    p50, p95, maximum = displacement_summary(boxes, result.boxes)
    return DbuEndpointCandidate(
        name=name,
        result=result,
        displacement_p50=p50,
        displacement_p95=p95,
        max_displacement=maximum,
    )


def project_dbu_endpoint_portfolio(
    boxes: np.ndarray,
    instance: FloorplanInstance,
    *,
    reference_boxes: np.ndarray,
    resolution: float = 0.001,
    legalization_time_limit_ms: int = 5000,
    projection_time_limit_ms: int = 10000,
    displacement_p95_fraction: float = 0.5,
) -> DbuEndpointPortfolioResult:
    """Try relation/contact routes only until a target-free-safe endpoint is found.

    ``reference_boxes`` is the candidate's own legalized layout before DBU
    materialization. It contains no target-derived information. The portfolio
    keeps selection separate from the official cost and always audits on the
    original instance.
    """

    incoming = np.asarray(boxes, dtype=np.float64)
    reference = np.asarray(reference_boxes, dtype=np.float64)
    if reference.shape != incoming.shape:
        raise ValueError("reference_boxes must match quantized boxes")
    if displacement_p95_fraction <= 0.0:
        raise ValueError("displacement_p95_fraction must be positive")
    span = max(
        float((incoming[:, 0] + incoming[:, 2]).max() - incoming[:, 0].min()),
        float((incoming[:, 1] + incoming[:, 3]).max() - incoming[:, 1].min()),
        1.0,
    )
    candidates: list[DbuEndpointCandidate] = []
    tried: list[str] = []

    def run(name: str, relation: str, contact: np.ndarray | None):
        tried.append(name)
        candidate = _candidate(
            name,
            incoming,
            instance,
            relation=relation,
            contact_reference=contact,
            resolution=resolution,
            legalization_time_limit_ms=legalization_time_limit_ms,
            projection_time_limit_ms=projection_time_limit_ms,
        )
        candidates.append(candidate)
        return candidate

    primary = run("expanded_continuous", "expanded", None)
    if (
        not primary.result.accepted
        or primary.displacement_p95 > displacement_p95_fraction * span
    ):
        restored = run("restored_continuous", "restored", None)
    else:
        restored = None
    accepted = [row for row in candidates if row.result.accepted]
    if not accepted:
        run("expanded_legalized", "expanded", reference)
        restored_raw = run("restored_legalized", "restored", reference)
        if restored_raw.result.accepted:
            restored = restored_raw
        accepted = [row for row in candidates if row.result.accepted]
    if accepted:
        selected = min(
            accepted,
            key=lambda row: (
                row.displacement_p95,
                row.max_displacement,
                row.name,
            ),
        )
    else:
        selected = min(
            candidates,
            key=lambda row: (
                row.result.overlap_after,
                row.result.grouping_after + row.result.boundary_after,
                row.name,
            ),
        )
    _ = restored
    return DbuEndpointPortfolioResult(
        selected=selected,
        candidates=tuple(candidates),
        tried_routes=tuple(tried),
    )


__all__ = [
    "DbuEndpointCandidate",
    "DbuEndpointPortfolioResult",
    "displacement_summary",
    "project_dbu_endpoint_portfolio",
]
