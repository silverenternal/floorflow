"""Fail-closed acceptance for geometry repairs using fpeval reports."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RepairDecision:
    accepted: bool
    reason: str
    deltas: dict[str, float]


def guarded_repair_decision(
    raw_report,
    candidate_report,
    *,
    epsilon: float = 1.0e-6,
) -> RepairDecision:
    """Accept only lower edit cost with no task-metric regression."""

    deltas = {
        "room_f1": candidate_report.rooms["f1"] - raw_report.rooms["f1"],
        "opening_f1": (
            candidate_report.openings["f1"] - raw_report.openings["f1"]
        ),
        "wall_f1": candidate_report.walls["f1"] - raw_report.walls["f1"],
        "validity": candidate_report.r_val - raw_report.r_val,
        "edit_cost": candidate_report.edit["total"] - raw_report.edit["total"],
    }
    regressions = [
        name
        for name in ("room_f1", "opening_f1", "wall_f1", "validity")
        if deltas[name] < -float(epsilon)
    ]
    if regressions:
        return RepairDecision(
            accepted=False,
            reason=f"regresses {', '.join(regressions)}",
            deltas=deltas,
        )
    if deltas["edit_cost"] >= -float(epsilon):
        return RepairDecision(
            accepted=False,
            reason="edit cost did not decrease",
            deltas=deltas,
        )
    return RepairDecision(accepted=True, reason="all guards passed", deltas=deltas)


def guarded_summary_decision(
    raw: dict[str, float],
    candidate: dict[str, float],
    *,
    epsilon: float = 1.0e-6,
) -> RepairDecision:
    deltas = {
        "room_f1": candidate["room_f1"] - raw["room_f1"],
        "opening_f1": candidate["opening_f1"] - raw["opening_f1"],
        "wall_f1": candidate["wall_f1"] - raw["wall_f1"],
        "validity": candidate["validity"] - raw["validity"],
        "edit_cost": candidate["edit_cost"] - raw["edit_cost"],
    }
    regressions = [
        name
        for name in ("room_f1", "opening_f1", "wall_f1", "validity")
        if deltas[name] < -float(epsilon)
    ]
    if regressions:
        return RepairDecision(
            accepted=False,
            reason=f"regresses {', '.join(regressions)}",
            deltas=deltas,
        )
    if deltas["edit_cost"] >= -float(epsilon):
        return RepairDecision(
            accepted=False,
            reason="edit cost did not decrease",
            deltas=deltas,
        )
    return RepairDecision(True, "all guards passed", deltas)
