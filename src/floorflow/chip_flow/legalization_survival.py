"""Measure whether raw candidate rankings survive legalization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations

import numpy as np
import torch

DEFAULT_VARIANTS = ("baseline", "expanded", "projector32")
METRICS = (
    "v_rel",
    "area_gap",
    "b2b_hpwl_gap",
    "p2b_hpwl_gap",
    "grouping",
    "boundary",
    "rho_mae",
)


def _stage_value(
    row: Mapping,
    variant: str,
    stage: str,
    metric: str,
) -> float | None:
    candidate = row.get(variant)
    if not isinstance(candidate, Mapping):
        return None
    stages = candidate.get("stages")
    if not isinstance(stages, Mapping):
        return None
    values = stages.get(stage)
    if not isinstance(values, Mapping) or metric not in values:
        return None
    value = float(values[metric])
    return value if np.isfinite(value) else None


def _relation(left: float, right: float, tolerance: float) -> int:
    delta = left - right
    if abs(delta) <= tolerance:
        return 0
    return -1 if delta < 0.0 else 1


def legalization_survival(
    rows: Sequence[Mapping],
    *,
    variants: Sequence[str] = DEFAULT_VARIANTS,
    tolerance: float = 1.0e-12,
) -> dict:
    """Compare all candidate pairs before and after legalization."""

    if len(variants) < 2:
        raise ValueError("legalization survival requires at least two variants")
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")
    result = {
        metric: {
            "defined": 0,
            "preserved": 0,
            "reversed": 0,
            "raw_tie": 0,
            "legalized_tie": 0,
            "top_choice_cases": 0,
            "top_choice_preserved": 0,
        }
        for metric in METRICS
    }
    seen_cases: set[int] = set()
    for row in rows:
        index = int(row["index"])
        seen_cases.add(index)
        for left, right in combinations(variants, 2):
            for metric in METRICS:
                raw_left = _stage_value(row, left, "refined", metric)
                raw_right = _stage_value(row, right, "refined", metric)
                legal_left = _stage_value(row, left, "legalized", metric)
                legal_right = _stage_value(row, right, "legalized", metric)
                values = (raw_left, raw_right, legal_left, legal_right)
                if any(value is None for value in values):
                    continue
                raw_relation = _relation(raw_left, raw_right, tolerance)
                legal_relation = _relation(legal_left, legal_right, tolerance)
                bucket = result[metric]
                if raw_relation == 0:
                    bucket["raw_tie"] += 1
                    continue
                if legal_relation == 0:
                    bucket["legalized_tie"] += 1
                    continue
                bucket["defined"] += 1
                if raw_relation == legal_relation:
                    bucket["preserved"] += 1
                else:
                    bucket["reversed"] += 1
        for metric in METRICS:
            raw_values = [
                _stage_value(row, variant, "refined", metric)
                for variant in variants
            ]
            legal_values = [
                _stage_value(row, variant, "legalized", metric)
                for variant in variants
            ]
            valid = [
                position
                for position, (raw, legal) in enumerate(
                    zip(raw_values, legal_values, strict=True)
                )
                if raw is not None and legal is not None
            ]
            if len(valid) < 2:
                continue
            raw_choice = min(valid, key=lambda position: raw_values[position])
            legal_choice = min(valid, key=lambda position: legal_values[position])
            bucket = result[metric]
            bucket["top_choice_cases"] += 1
            if raw_choice == legal_choice:
                bucket["top_choice_preserved"] += 1
    for bucket in result.values():
        defined = bucket["defined"]
        top_cases = bucket["top_choice_cases"]
        bucket["pairwise_lsr"] = (
            bucket["preserved"] / defined if defined else None
        )
        bucket["top_choice_retention"] = (
            bucket["top_choice_preserved"] / top_cases if top_cases else None
        )
    defined_total = sum(bucket["defined"] for bucket in result.values())
    preserved_total = sum(bucket["preserved"] for bucket in result.values())
    top_total = sum(bucket["top_choice_cases"] for bucket in result.values())
    top_preserved = sum(
        bucket["top_choice_preserved"] for bucket in result.values()
    )
    return {
        "version": "chip-legalization-survival-v1",
        "status": "complete" if defined_total else "insufficient_data",
        "cases": len(seen_cases),
        "variants": list(variants),
        "metrics": result,
        "summary": {
            "defined_pairs": defined_total,
            "preserved_pairs": preserved_total,
            "pairwise_lsr": (
                preserved_total / defined_total if defined_total else None
            ),
            "top_choice_cases": top_total,
            "top_choice_retention": (
                top_preserved / top_total if top_total else None
            ),
        },
    }


def legalization_rank_retention_loss(
    predicted_scores: torch.Tensor,
    legalized_scores: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Distill lower-is-better legalized rankings into raw-stage scores."""

    if predicted_scores.shape != legalized_scores.shape:
        raise ValueError("predicted and legalized scores must match")
    if predicted_scores.ndim != 2 or predicted_scores.shape[1] < 2:
        raise ValueError("rank retention requires [batch, candidates] scores")
    if temperature <= 0.0:
        raise ValueError("rank distillation temperature must be positive")
    if mask is None:
        mask = torch.ones_like(predicted_scores, dtype=torch.bool)
    if mask.shape != predicted_scores.shape:
        raise ValueError("rank retention mask must match scores")
    mask = mask.bool()
    if not bool(mask.any(dim=-1).all()):
        raise ValueError("each case requires at least one legalized candidate")
    target = torch.softmax(
        (-legalized_scores.float() / float(temperature)).masked_fill(
            ~mask,
            float("-inf"),
        ),
        dim=-1,
    )
    log_probability = torch.log_softmax(
        (-predicted_scores.float()).masked_fill(~mask, float("-inf")),
        dim=-1,
    )
    return -(target * log_probability).sum(dim=-1).mean()


__all__ = [
    "DEFAULT_VARIANTS",
    "METRICS",
    "legalization_rank_retention_loss",
    "legalization_survival",
]
