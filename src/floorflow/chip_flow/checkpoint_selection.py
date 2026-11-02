"""Predeclared legalized checkpoint ranking and shortlist screening."""

from __future__ import annotations

from collections.abc import Iterable


def legalized_candidate_key(row: dict) -> tuple[float, ...]:
    """Rank feasible legalized endpoints before raw-surrogate quality."""

    feasible = 0.0 if bool(row.get("legalized_feasible", False)) else 1.0
    return (
        feasible,
        float(row.get("legalized_hard_cost", float("inf"))),
        float(row.get("legalized_v_rel", float("inf"))),
        float(row.get("legalized_area_gap", float("inf"))),
        float(row.get("legalized_b2b_hpwl_gap", float("inf"))),
        float(row.get("legalized_grouping", float("inf"))),
        float(row.get("legalized_boundary", float("inf"))),
        float(row.get("legalized_displacement_mean_um", float("inf"))),
        int(row.get("legalizer_rung", 5)),
        float(row.get("raw_total", float("inf"))),
        int(row["step"]),
    )


def rank_legalized_candidates(rows: Iterable[dict]) -> list[dict]:
    return sorted(rows, key=legalized_candidate_key)


def select_legalized_candidate(rows: Iterable[dict]) -> dict:
    ranked = rank_legalized_candidates(rows)
    if not ranked:
        raise ValueError("no legalized checkpoint candidates are available")
    return ranked[0]


__all__ = [
    "legalized_candidate_key",
    "rank_legalized_candidates",
    "select_legalized_candidate",
]
