"""Contract tests for the official SOTA baseline feasibility lock."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict:
    return json.loads(
        (ROOT / "configs/sota_official_baselines_v2.json").read_text(
            encoding="utf-8"
        )
    )


def test_all_planned_external_methods_have_an_audited_state() -> None:
    payload = _config()
    rows = {row["id"]: row for row in payload["baselines"]}
    assert set(rows) == {
        "ce2eplan",
        "raster2seq",
        "housemind",
        "tlc_plan",
        "fmlm",
        "renolayout",
    }
    assert rows["ce2eplan"]["status"] == "executable_verified"
    assert rows["raster2seq"]["status"] == "executable_verified"
    assert rows["housemind"]["status"] == "weights_only_blocked"


def test_executable_baselines_are_pinned_and_use_gpu_zero() -> None:
    payload = _config()
    assert payload["execution_device"] == "cuda:0"
    for row in payload["baselines"]:
        if row["status"] != "executable_verified":
            continue
        assert len(row["commit"]) == 40
        assert row["matched_comparison"]["eligible"] is False
        assert row["execution_device"] == "cuda:0"


def test_ce2eplan_contract_preserves_dataset_and_metric_boundaries() -> None:
    row = next(item for item in _config()["baselines"] if item["id"] == "ce2eplan")
    assert row["dataset"]["cases"] == 12002
    assert row["published_anchor"]["FID"] == 1.08
    assert row["license"] == "not_released"
    assert "not an editing baseline" in row["claim_boundary"]


def test_raster2seq_does_not_hide_the_case_unit_mismatch() -> None:
    row = next(item for item in _config()["baselines"] if item["id"] == "raster2seq")
    assert row["dataset"]["cases"] == 509
    assert row["case_unit"]["floorflow_floor_cases"] == 503
    assert row["case_unit"]["one_to_one_bijection_available"] is False
    assert row["matched_comparison"]["eligible"] is False


def test_housemind_partial_reproduction_cannot_become_a_metric_row() -> None:
    row = next(item for item in _config()["baselines"] if item["id"] == "housemind")
    assert row["partial_reproduction"]["eligible_for_metrics"] is False
    assert len(row["missing_contract"]) >= 5
    assert "no HouseMind-E output" in row["claim_boundary"]


def test_oracle_control_capacity_is_explicitly_non_deployable() -> None:
    boundary = _config()["oracle_control_boundary"]
    assert boundary["room_macro_iou"] == 0.9718283840634568
    assert "target-derived" in boundary["interpretation"]
    assert "deployable reconstruction" in boundary["forbidden_use"]


def test_baseline_audit_script_respects_file_length_limit() -> None:
    script = ROOT / "scripts/audit_official_sota_baselines.py"
    assert len(script.read_text(encoding="utf-8").splitlines()) < 600
