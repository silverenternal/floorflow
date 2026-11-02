"""Val-only candidate selection with fail-closed rollback."""

from __future__ import annotations

from dataclasses import dataclass


NON_REGRESSION_GUARDS = {
    "room_macro_iou": 0.002,
    "instance_room_f1": 0.005,
    "fpeval_room_f1": 0.010,
    "conditional_watertight": 0.005,
}

IMPROVEMENT_METRICS = (
    "opening_f1",
    "furniture_iou",
    "wall_iou",
    "fpeval_wall_f1",
)


@dataclass(frozen=True)
class SelectionDecision:
    eligible: bool
    reasons: tuple[str, ...]
    improvements: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "eligible": self.eligible,
            "reasons": list(self.reasons),
            "improvements": list(self.improvements),
        }


def validate_metric_report(report: dict) -> None:
    if report.get("best_of_n_used", True):
        raise ValueError("val candidate reports must not use best-of-N")
    if not isinstance(report.get("metrics"), dict):
        raise ValueError("report is missing metrics")


def evaluate_candidate(
    reference: dict,
    candidate: dict,
) -> SelectionDecision:
    """Require primary non-regression and at least one real overlay gain."""

    validate_metric_report(reference)
    validate_metric_report(candidate)
    if reference.get("cases") != candidate.get("cases"):
        return SelectionDecision(False, ("case_count_mismatch",), ())
    reference_metrics = reference["metrics"]
    candidate_metrics = candidate["metrics"]
    reasons = []
    for name, tolerance in NON_REGRESSION_GUARDS.items():
        baseline = reference_metrics.get(name)
        value = candidate_metrics.get(name)
        if baseline is None or value is None:
            reasons.append(f"missing_{name}")
        elif float(value) < float(baseline) - float(tolerance):
            reasons.append(f"regressed_{name}")
    improvements = tuple(
        name
        for name in IMPROVEMENT_METRICS
        if name in reference_metrics
        and name in candidate_metrics
        and float(candidate_metrics[name]) > float(reference_metrics[name]) + 0.005
    )
    if not improvements:
        reasons.append("no_overlay_improvement")
    return SelectionDecision(
        eligible=not reasons,
        reasons=tuple(reasons),
        improvements=improvements,
    )


__all__ = [
    "IMPROVEMENT_METRICS",
    "NON_REGRESSION_GUARDS",
    "SelectionDecision",
    "evaluate_candidate",
    "validate_metric_report",
]
