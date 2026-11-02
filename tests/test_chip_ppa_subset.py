from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_chip_ppa_subset",
    ROOT / "scripts/run_chip_ppa_subset.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
aggregate = MODULE.aggregate


def _run(
    arm: str,
    wirelength: float,
    drc: int = 0,
    *,
    case: str = "nangate45/example",
    seed: int = 19,
) -> dict:
    return {
        "case": case,
        "arm": arm,
        "seed": seed,
        "status": "completed",
        "metrics_values": {
            "detailedroute__route__wirelength": wirelength,
            "detailedroute__route__drc_errors": drc,
            "globalplace__gpl__routability__congestion": 1.0,
            "finish__timing__setup__ws": -0.1,
            "finish__power__total": 1.0,
            "finish__design__instance__area__stdcell": 1.0,
        },
    }


def test_aggregate_applies_frozen_promotion_gate() -> None:
    summary = aggregate([_run("candidate", 0.97), _run("control", 1.0)], [19])
    assert summary["summary"]["primary_relative_wirelength_mean"] == pytest.approx(-0.03)
    assert summary["promotion_gate"] == "pass"


def test_aggregate_rejects_drc_regression() -> None:
    summary = aggregate(
        [_run("candidate", 0.9, drc=1), _run("control", 1.0, drc=0)],
        [19],
    )
    assert summary["checks"]["drc_zero_both_arms"] is False
    assert summary["promotion_gate"] == "reject_ppa_claim"


def test_aggregate_reports_seed_and_case_sensitivity() -> None:
    runs = []
    for case, effects in (
        ("nangate45/a", (-0.03, -0.01)),
        ("nangate45/b", (-0.05, -0.03)),
    ):
        for seed, effect in zip((19, 20), effects):
            runs.extend(
                [
                    _run(
                        "candidate",
                        1.0 + effect,
                        case=case,
                        seed=seed,
                    ),
                    _run("control", 1.0, case=case, seed=seed),
                ]
            )
    summary = aggregate(runs, [19, 20])
    assert summary["wirelength_win_tie_loss"] == {
        "wins": 4,
        "ties": 0,
        "losses": 0,
    }
    assert summary["case_relative_wirelength_means"]["nangate45/a"] == pytest.approx(-0.02)
    assert summary["case_relative_wirelength_means"]["nangate45/b"] == pytest.approx(-0.04)
    assert summary["leave_one_case_out"]["nangate45/a"] == pytest.approx(-0.04)
    assert summary["leave_one_case_out"]["nangate45/b"] == pytest.approx(-0.02)


def test_secondary_gate_uses_congestion_or_timing() -> None:
    candidate = _run("candidate", 0.97)
    control = _run("control", 1.0)
    candidate["metrics_values"]["globalplace__gpl__routability__congestion"] = 1.01
    candidate["metrics_values"]["finish__timing__setup__ws"] = -0.07
    control["metrics_values"]["globalplace__gpl__routability__congestion"] = 1.0
    control["metrics_values"]["finish__timing__setup__ws"] = -0.04
    summary = aggregate([candidate, control], [19])
    assert summary["checks"]["congestion_mean_at_most_2pct"] is True
    assert summary["checks"]["timing_ws_mean_nonregression_20ps"] is False
    assert summary["promotion_gate"] == "pass"


def test_aggregate_cannot_ignore_declared_missing_cases() -> None:
    summary = aggregate(
        [_run("candidate", 0.97), _run("control", 1.0)],
        [19],
        ["nangate45/example", "nangate45/missing"],
    )

    assert summary["checks"]["all_pairs_completed"] is False
    assert summary["promotion_gate"] == "reject_ppa_claim"
