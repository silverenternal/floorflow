from __future__ import annotations

import json
from pathlib import Path

import pytest

from floorflow.evaluation.atom_proxy_consistency import (
    ConsistencyThresholds,
    build_consistency_report,
    evaluate_measurement,
    load_measurement_payloads,
)


def _payload(baseline: str = "c0") -> dict:
    rows = []
    checkpoints = [f"c{index}" for index in range(8)]
    for case_index in range(4):
        for checkpoint_index, checkpoint in enumerate(checkpoints):
            rows.append(
                {
                    "seed": 19,
                    "checkpoint": checkpoint,
                    "case_id": f"case-{case_index}",
                    "proxy": checkpoint_index + case_index * 0.01,
                    "canonical": checkpoint_index + case_index * 0.02,
                }
            )
    return {
        "id": "a1-proxy",
        "atom_id": "A1",
        "proxy_metric": "proxy",
        "canonical_metric": "canonical",
        "baseline_checkpoint": baseline,
        "target_used_at_inference": False,
        "rows": rows,
        "sensitivity": [
            {
                "case_id": "attack",
                "attack": "degrade",
                "proxy_baseline": 2.0,
                "proxy_attack": 1.0,
                "canonical_baseline": 2.0,
                "canonical_attack": 1.0,
            }
        ],
    }


def test_consistent_measurement_passes_all_checks(tmp_path: Path) -> None:
    path = tmp_path / "measurement.json"
    path.write_text(json.dumps(_payload()), encoding="utf-8")
    measurement = load_measurement_payloads([path])[0]
    result = evaluate_measurement(
        measurement,
        ConsistencyThresholds(
            spearman_rho_min=-1.0,
            pearson_rho_min=-1.0,
            top_checkpoint_agreement_min=0.9,
            paired_sign_agreement_min=0.8,
            false_pass_rate_max=0.05,
            false_fail_rate_max=0.1,
        ),
    )
    assert result["passed"]
    assert result["summary"]["spearman_rho"] == pytest.approx(1.0)
    assert result["summary"]["sensitivity_rate"] == 1.0


def test_flat_measurement_is_rejected(tmp_path: Path) -> None:
    payload = _payload()
    for row in payload["rows"]:
        row["proxy"] = 1.0
    path = tmp_path / "flat.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    measurement = load_measurement_payloads([path])[0]
    result = evaluate_measurement(measurement, ConsistencyThresholds())
    assert result["summary"]["flat_proxy"]
    assert not result["passed"]


def test_missing_sensitivity_and_coverage_fail_closed(tmp_path: Path) -> None:
    payload = _payload()
    payload["sensitivity"] = []
    payload["rows"] = [
        row
        for row in payload["rows"]
        if int(row["checkpoint"].removeprefix("c")) < 4
    ]
    path = tmp_path / "open.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    report = build_consistency_report(
        load_measurement_payloads([path]),
        ConsistencyThresholds(),
    )
    result = report["measurements"][0]
    assert not result["checks"]["sensitivity"]
    assert not result["checks"]["checkpoint_coverage"]
    assert report["summary"] == {"measurements": 1, "passed": 0, "failed": 1}


def test_duplicate_measurement_ids_are_rejected(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps(_payload()), encoding="utf-8")
    second.write_text(json.dumps(_payload()), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate measurement"):
        load_measurement_payloads([first, second])


def test_cross_seed_case_mismatch_fails_same_case_gate(tmp_path: Path) -> None:
    payload = _payload()
    original = payload["rows"]
    payload["rows"] = [
        row for row in original if row["case_id"] != "case-3"
    ] + [dict(row, seed=23) for row in original]
    path = tmp_path / "cross-seed.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    result = evaluate_measurement(
        load_measurement_payloads([path])[0],
        ConsistencyThresholds(),
    )
    assert not result["summary"]["same_cases_across_seeds"]
    assert not result["summary"]["same_cases_all_seeds"]
    assert not result["checks"]["same_cases"]
