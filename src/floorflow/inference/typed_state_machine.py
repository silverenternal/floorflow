"""Verified typed-state candidate commit and rollback."""

from __future__ import annotations

from collections.abc import Callable, Mapping


def select_verified_state(
    candidates: Mapping[str, dict],
    verifier: Callable[[dict], float],
    *,
    fallback: str,
    minimum_improvement: float = 0.0,
) -> tuple[dict, dict[str, float | str]]:
    scores = {
        name: float(verifier(candidate))
        for name, candidate in candidates.items()
    }
    return select_scored_state(
        candidates,
        scores,
        fallback=fallback,
        minimum_improvement=minimum_improvement,
    )


def select_scored_state(
    candidates: Mapping[str, dict],
    scores: Mapping[str, float],
    *,
    fallback: str,
    minimum_improvement: float = 0.0,
) -> tuple[dict, dict[str, float | str]]:
    if fallback not in candidates:
        raise ValueError("fallback candidate is missing")
    scores = {name: float(scores[name]) for name in candidates}
    best = max(scores, key=scores.get)
    if scores[best] < scores[fallback] + float(minimum_improvement):
        best = fallback
    return candidates[best], {
        "selected": best,
        **{f"score_{name}": score for name, score in scores.items()},
    }


__all__ = ["select_scored_state", "select_verified_state"]
