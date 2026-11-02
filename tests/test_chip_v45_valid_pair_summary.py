from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "chip_v45_valid_pair_summary",
    ROOT / "scripts/build_chip_v45_valid_pair_summary.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _row(case: str, wire: float, congestion: float, setup_candidate: float) -> dict:
    return {
        "case": case,
        "source_family": case.split("/", maxsplit=1)[-1],
        "seed": 19,
        "detailedroute__route__wirelength": {"relative": wire},
        "globalplace__gpl__routability__congestion": {"relative": congestion},
        "finish__timing__setup__ws": {
            "candidate": setup_candidate,
            "control": 0.0,
        },
        "finish__power__total": {"relative": 0.01},
        "finish__design__instance__area__stdcell": {"relative": 0.02},
        "candidate_drc": 0,
        "control_drc": 0,
    }


def test_valid_pair_summary_excludes_implicit_ties_and_reports_tail() -> None:
    rows = [
        _row("p/win", -0.05, -0.01, -0.1),
        _row("p/loss", 0.10, 0.02, -0.2),
    ]

    result = MODULE.summarize(rows)

    assert result["routed_wirelength_relative"]["mean"] == pytest.approx(0.025)
    assert result["wirelength_wins_ties_losses"] == {
        "wins": 1,
        "ties": 0,
        "losses": 1,
    }
    assert result["finish_setup_ws_delta_ns"]["min"] == pytest.approx(-0.2)
    assert result["drc_zero"]
    per_case = MODULE.per_case_rows(rows)
    assert [row["case"] for row in per_case] == ["p/win", "p/loss"]
    assert per_case[0]["routed_wirelength"]["relative"] == pytest.approx(-0.05)
