"""Generate-validate-repair loop for typed floorplan states."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RepairStep:
    step: int
    accepted: bool
    score: float
    best_score: float


@dataclass
class RepairLoopResult:
    state: Any
    score: float
    trace: list[RepairStep] = field(default_factory=list)


def run_verified_repair_loop(
    initial_state: Any,
    propose: Callable[[Any, int], Any | None],
    verify: Callable[[Any], float],
    *,
    max_steps: int = 3,
    minimum_improvement: float = 0.0,
) -> RepairLoopResult:
    """Accept only verifier-approved typed state edits and keep the best state."""

    best_state = initial_state
    best_score = float(verify(initial_state))
    trace = []
    for step in range(int(max_steps)):
        candidate = propose(best_state, step)
        if candidate is None:
            trace.append(
                RepairStep(step, False, best_score, best_score)
            )
            continue
        score = float(verify(candidate))
        accepted = score >= best_score + float(minimum_improvement)
        if accepted:
            best_state = candidate
            best_score = score
        trace.append(RepairStep(step, accepted, score, best_score))
    return RepairLoopResult(
        state=best_state,
        score=best_score,
        trace=trace,
    )


__all__ = ["RepairLoopResult", "RepairStep", "run_verified_repair_loop"]
