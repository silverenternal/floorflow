"""Target-free selection among legalization candidates."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median


@dataclass(frozen=True)
class LegalizerCandidate:
    name: str
    v_rel: float
    displacement: float
    boundary: float
    grouping: float
    rung: int
    solves: int
    b2b_hpwl: float | None = None
    p2b_hpwl: float | None = None
    outline_compactness: float | None = None
    outline_aspect: float | None = None


def select_legalizer_candidate(
    baseline: LegalizerCandidate,
    candidates: tuple[LegalizerCandidate, ...],
    *,
    policy: str = "legality_first",
    vrel_tolerance: float = 0.01,
) -> LegalizerCandidate:
    """Choose legality first and reject any solver-work escalation."""

    eligible = (
        candidate
        for candidate in (baseline, *candidates)
        if candidate.rung <= baseline.rung
        and candidate.solves <= baseline.solves
    )
    eligible = tuple(eligible)
    if policy == "legality_first":
        return min(eligible, key=_legality_key)
    if policy in ("geometry_consensus", "legality_guarded_consensus"):
        complete = tuple(candidate for candidate in eligible if _has_geometry(candidate))
        if not complete:
            return min(eligible, key=_legality_key)
        if policy == "legality_guarded_consensus":
            legality_first = min(eligible, key=_legality_key)
            if _has_geometry(legality_first) and not _geometry_risk(
                legality_first,
                complete,
            ):
                return legality_first
        return min(complete, key=lambda candidate: _consensus_key(candidate, complete))
    if policy != "balanced":
        raise ValueError("unknown legalizer selection policy")
    best_v_rel = min(candidate.v_rel for candidate in eligible)
    protected = tuple(
        candidate
        for candidate in eligible
        if candidate.v_rel <= best_v_rel + float(vrel_tolerance)
    )
    if not any(
        candidate.b2b_hpwl is not None
        and candidate.outline_compactness is not None
        for candidate in protected
    ):
        return min(protected, key=_legality_key)
    return min(protected, key=_balanced_key)


def _legality_key(candidate: LegalizerCandidate) -> tuple:
    return (
        candidate.v_rel,
        candidate.displacement,
        candidate.boundary,
        candidate.grouping,
        candidate.name,
    )


def _balanced_key(candidate: LegalizerCandidate) -> tuple:
    geometry = (
        float(candidate.b2b_hpwl or 0.0) + float(candidate.p2b_hpwl or 0.0)
    )
    compactness_penalty = 1.0 - float(candidate.outline_compactness or 0.0)
    aspect_penalty = abs(float(candidate.outline_aspect or 0.0))
    return (
        geometry,
        compactness_penalty,
        aspect_penalty,
        *_legality_key(candidate),
    )


def _has_geometry(candidate: LegalizerCandidate) -> bool:
    return (
        candidate.b2b_hpwl is not None
        and candidate.p2b_hpwl is not None
        and candidate.outline_compactness is not None
        and candidate.outline_aspect is not None
    )


def _consensus_key(
    candidate: LegalizerCandidate,
    candidates: tuple[LegalizerCandidate, ...],
) -> tuple:
    b2b_values = sorted(float(item.b2b_hpwl) for item in candidates)
    p2b_values = sorted(float(item.p2b_hpwl) for item in candidates)
    compactness_values = sorted(
        -float(item.outline_compactness) for item in candidates
    )
    aspect_values = sorted(abs(float(item.outline_aspect)) for item in candidates)
    rank = (
        _rank(candidate.v_rel, sorted(item.v_rel for item in candidates))
        + _rank(float(candidate.b2b_hpwl), b2b_values)
        + _rank(float(candidate.p2b_hpwl), p2b_values)
        + _rank(-float(candidate.outline_compactness), compactness_values)
        + _rank(abs(float(candidate.outline_aspect)), aspect_values)
    )
    return (rank, *_legality_key(candidate))


def _rank(value: float, sorted_values: list[float]) -> int:
    return min(
        index
        for index, candidate in enumerate(sorted_values)
        if float(value) <= float(candidate) + 1.0e-12
    )


def _geometry_risk(
    candidate: LegalizerCandidate,
    candidates: tuple[LegalizerCandidate, ...],
) -> bool:
    median_b2b = median(float(item.b2b_hpwl) for item in candidates)
    median_p2b = median(float(item.p2b_hpwl) for item in candidates)
    return (
        float(candidate.outline_compactness) < 0.65
        or abs(float(candidate.outline_aspect)) > 0.75
        or float(candidate.b2b_hpwl) > 1.25 * median_b2b
        or float(candidate.p2b_hpwl) > 1.25 * median_p2b
    )


__all__ = ["LegalizerCandidate", "select_legalizer_candidate"]
