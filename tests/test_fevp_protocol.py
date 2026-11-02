"""Tests for the FEVP v1.0 schema and semantic validator."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from floorflow.evaluation.fevp_protocol import (
    FEVPValidationError,
    aggregate_case_rows,
    load_json,
    validate_case_row,
    validate_report,
)

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples/fevp"


def _complete() -> dict:
    return load_json(EXAMPLES / "complete_report_v1.json")


def test_schema_files_are_valid_json() -> None:
    for name in ("fevp_report_v1.schema.json", "fevp_case_row_v1.schema.json"):
        payload = json.loads((ROOT / "schema" / name).read_text(encoding="utf-8"))
        assert payload["$schema"].endswith("2020-12/schema")


def test_complete_report_passes_strict_validation() -> None:
    assert validate_report(_complete(), require_complete=True) == []


def test_incomplete_report_is_valid_but_not_complete() -> None:
    report = load_json(EXAMPLES / "incomplete_report_v1.json")

    assert validate_report(report) == []
    errors = validate_report(report, require_complete=True)
    assert any("report is incomplete" in error for error in errors)


def test_reported_metric_requires_value_and_source() -> None:
    report = copy.deepcopy(_complete())
    report["mandatory"]["validity"].pop("value")
    report["mandatory"]["validity"].pop("source")

    errors = validate_report(report)

    assert any("validity.value" in error for error in errors)
    assert any("validity.source" in error for error in errors)


def test_unreported_metric_cannot_carry_imputed_value() -> None:
    report = load_json(EXAMPLES / "incomplete_report_v1.json")
    report["mandatory"]["edge_overlap"]["value"] = 0.5

    errors = validate_report(report)

    assert any("must not carry a value" in error for error in errors)


def test_case_rows_aggregate_conditional_watertight_denominator() -> None:
    payload = load_json(EXAMPLES / "case_rows_v1.json")
    rows = payload["rows"]

    assert all(validate_case_row(row) == [] for row in rows)
    summary = aggregate_case_rows(rows)
    assert summary["cases"] == 2
    assert summary["watertight_eligible_cases"] == 1
    assert summary["conditional_watertightness"] == 1.0
    assert summary["optional"]["delta_iou"] == pytest.approx(0.97)


def test_case_row_rejects_watertight_value_for_nonwatertight_target() -> None:
    row = load_json(EXAMPLES / "case_rows_v1.json")["rows"][1]
    row["conditional_watertightness"] = 1.0

    errors = validate_case_row(row)

    assert any("must be null" in error for error in errors)


def test_assert_valid_report_raises_for_required_completeness() -> None:
    from floorflow.evaluation.fevp_protocol import assert_valid_report

    report = load_json(EXAMPLES / "incomplete_report_v1.json")
    with pytest.raises(FEVPValidationError):
        assert_valid_report(report, require_complete=True)
