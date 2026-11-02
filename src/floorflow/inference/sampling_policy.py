"""Auditable sampling budgets for conditional floorplan generation.

Best-of-N is a maximum-order statistic, not an estimate of the model
distribution. This module keeps candidate counts explicit and separates
single-sample evaluation, distribution estimation, and read-only ceiling
diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


SamplingPurpose = Literal["single_sample", "distribution", "diagnostic_ceiling"]


@dataclass(frozen=True)
class SamplingPolicy:
    purpose: SamplingPurpose = "single_sample"
    candidates_per_condition: int = 1
    seed: int = 0
    diagnostic_limit: int = 16

    def __post_init__(self) -> None:
        if self.candidates_per_condition < 1:
            raise ValueError("candidates_per_condition must be positive")
        if self.purpose == "single_sample" and self.candidates_per_condition != 1:
            raise ValueError("single_sample evaluation must use exactly one candidate")
        if self.purpose == "distribution" and self.candidates_per_condition < 2:
            raise ValueError("distribution evaluation needs at least two candidates")
        if self.purpose == "diagnostic_ceiling" and self.candidates_per_condition > self.diagnostic_limit:
            raise ValueError("ceiling diagnostic exceeds its configured candidate budget")

    @property
    def primary_metric_eligible(self) -> bool:
        return self.purpose in {"single_sample", "distribution"}

    @property
    def uses_best_of_n(self) -> bool:
        return self.purpose == "diagnostic_ceiling"

    def manifest(self) -> dict[str, str | int | bool]:
        return {
            "purpose": self.purpose,
            "candidates_per_condition": self.candidates_per_condition,
            "seed": self.seed,
            "primary_metric_eligible": self.primary_metric_eligible,
            "best_of_n_used": self.uses_best_of_n,
            "selection_rule": "none" if not self.uses_best_of_n else "maximum_for_diagnostic_only",
            "target_used_for_selection": False,
        }


def first_sample(samples):
    """Return the seed-controlled canonical sample without metric selection."""

    if len(samples) == 0:
        raise ValueError("at least one sample is required")
    return samples[0]


def expected_sample_statistics(values: np.ndarray) -> dict[str, float]:
    """Summarise every sample without selecting an extremum."""

    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        raise ValueError("values must not be empty")
    flat = values.ravel()
    return {
        "mean": float(flat.mean()),
        "median": float(np.median(flat)),
        "standard_error": float(flat.std(ddof=1) / np.sqrt(flat.size)) if flat.size > 1 else 0.0,
        "worst": float(flat.min()),
    }


def best_of_n_curve(values: np.ndarray, n_values: tuple[int, ...] = (1, 2, 4, 8)) -> dict[int, float]:
    """Compute a read-only oracle ceiling for fixed, predeclared budgets."""

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("values must have shape [conditions, candidates]")
    available = values.shape[1]
    budgets = tuple(sorted({min(max(1, int(n)), available) for n in n_values}))
    return {budget: float(values[:, :budget].max(axis=1).mean()) for budget in budgets}


def policy_from_manifest(manifest: dict) -> SamplingPolicy:
    """Validate sampling metadata before it enters a report or gate."""

    return SamplingPolicy(
        purpose=manifest.get("purpose", "single_sample"),
        candidates_per_condition=int(manifest.get("candidates_per_condition", 1)),
        seed=int(manifest.get("seed", 0)),
    )


def validate_primary_report(report: dict) -> tuple[bool, list[str]]:
    """Fail closed when a selection-based number is presented as primary."""

    errors = []
    if report.get("best_of_n_used", False):
        errors.append("best_of_n_used must be false for a primary report")
    manifest = report.get("sampling_policy")
    if not isinstance(manifest, dict):
        errors.append("sampling_policy manifest is missing")
        return False, errors
    try:
        policy = policy_from_manifest(manifest)
    except (TypeError, ValueError) as error:
        return False, [*errors, str(error)]
    if not policy.primary_metric_eligible:
        errors.append("sampling purpose is diagnostic-only")
    if manifest.get("selection_rule") != "none":
        errors.append("primary reports must not use sample selection")
    if manifest.get("target_used_for_selection", False):
        errors.append("target-informed selection is forbidden")
    return not errors, errors
