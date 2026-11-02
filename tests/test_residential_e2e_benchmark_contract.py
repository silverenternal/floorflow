"""Tests for the residential end-to-end benchmark claim matrix."""

from __future__ import annotations

from pathlib import Path

from floorflow.evaluation.e2e_benchmark_contract import (
    load_benchmark_config,
    validate_benchmark_config,
    validate_claim_rows,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/residential_e2e_benchmark_v1.json"


def test_benchmark_config_is_residential_and_fail_closed() -> None:
    config = load_benchmark_config(CONFIG)
    assert validate_benchmark_config(config) == []
    assert config["scope"] == "residential_floorplan_only"
    assert config["engineering_guardrails"]["do_not_touch_chips"] is True
    assert config["claim_rules"]["forbidden_claims"]


def test_unmatched_methods_cannot_enable_claims() -> None:
    config = load_benchmark_config(CONFIG)
    rows = [
        {
            "method": "Raster2Seq",
            "track": "cubicasa_reconstruction",
            "claim_state": "matched",
            "executable": True,
            "same_scorer": False,
        },
        {
            "method": "CE2EPlan",
            "track": "cubicasa_predicted_control_editing",
            "claim_state": "context_only",
            "executable": True,
            "same_scorer": False,
        },
    ]
    errors = validate_claim_rows(config, rows)
    assert any("same scorer" in error for error in errors)
    assert any("CE2EPlan cannot enter an editing" in error for error in errors)


def test_only_matched_executable_rows_can_be_leaderboard_rows() -> None:
    config = load_benchmark_config(CONFIG)
    rows = [
        {
            "method": "Raster2Seq",
            "track": "cubicasa_reconstruction",
            "claim_state": "context_only",
            "executable": True,
            "same_scorer": False,
        }
    ]
    assert validate_claim_rows(config, rows) == []
