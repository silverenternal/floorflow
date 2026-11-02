"""Tests for the frozen CAGN five-arm report contract."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_five_arm_report_has_exact_hard_masks_and_matched_parameters() -> None:
    report = json.loads(
        (ROOT / "outputs/reports/residential_cagn_five_arm_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["status"] == "passed"
    assert len(report["arms"]) == 5
    assert report["checks"]["hard_mask_exact"]
    assert report["checks"]["parameter_matched"]
    assert report["paired_comparisons"]["full_minus_hard"]["ci_low"] > 0.0
