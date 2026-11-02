from __future__ import annotations

from floorflow.training.val_selection import evaluate_candidate


def _report(room: float, fpeval_room: float, opening: float) -> dict:
    return {
        "cases": 32,
        "best_of_n_used": False,
        "metrics": {
            "room_macro_iou": room,
            "instance_room_f1": 0.5,
            "fpeval_room_f1": fpeval_room,
            "conditional_watertight": 0.95,
            "opening_f1": opening,
            "furniture_iou": 0.6,
            "wall_iou": 0.7,
            "fpeval_wall_f1": 0.8,
        },
    }


def test_candidate_regression_rolls_back() -> None:
    decision = evaluate_candidate(
        _report(0.70, 0.50, 0.80),
        _report(0.69, 0.45, 0.84),
    )
    assert not decision.eligible
    assert "regressed_room_macro_iou" in decision.reasons
    assert "regressed_fpeval_room_f1" in decision.reasons


def test_candidate_requires_overlay_improvement() -> None:
    decision = evaluate_candidate(
        _report(0.70, 0.50, 0.80),
        _report(0.701, 0.501, 0.801),
    )
    assert not decision.eligible
    assert "no_overlay_improvement" in decision.reasons


def test_non_regressing_overlay_candidate_is_eligible() -> None:
    decision = evaluate_candidate(
        _report(0.70, 0.50, 0.80),
        _report(0.701, 0.501, 0.82),
    )
    assert decision.eligible
    assert decision.improvements == ("opening_f1",)
