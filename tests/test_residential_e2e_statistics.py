"""Tests for five-seed intervals and Holm correction."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "residential_e2e_multiseed_report",
    ROOT / "scripts/build_residential_e2e_multiseed_report.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_student_t_interval_reports_range_and_ci() -> None:
    interval = MODULE._interval([1.0, 2.0, 3.0, 4.0, 5.0])
    assert interval["mean"] == 3.0
    assert interval["min"] == 1.0
    assert interval["max"] == 5.0
    assert interval["ci_low"] < 3.0 < interval["ci_high"]


def test_holm_correction_is_monotone_and_preserves_order() -> None:
    results = {
        "a": {"p_value": 0.01},
        "b": {"p_value": 0.04},
        "c": {"p_value": 0.03},
    }
    MODULE._holm(results)
    assert results["a"]["holm_adjusted_p"] < results["c"]["holm_adjusted_p"]
    assert results["c"]["holm_adjusted_p"] == results["b"]["holm_adjusted_p"]
    assert all("holm_significant" in values for values in results.values())
