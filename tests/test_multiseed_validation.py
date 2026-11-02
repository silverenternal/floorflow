"""Tests for fail-closed multi-seed hard-gate validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from floorflow.evaluation.multiseed_validation import build_multiseed_report


def write_run(root: Path, seed: int, failures: int = 0) -> Path:
    rows = []
    for index in range(503):
        failed = index < failures
        rows.append(
            {
                "name": f"case-{index:04d}",
                "fpeval_validity": 1.0,
                "fpeval_watertight": 0.0 if failed else 1.0,
                "target_watertight": 1.0,
                "room_target_recall": 0.90 if failed else 1.0,
                "outside_footprint_pixels": 0.0,
            }
        )
    metrics_path = root / f"seed{seed}" / "metrics.json"
    metrics_path.parent.mkdir(parents=True)
    metrics_path.write_text(
        json.dumps({"seed": 23, "metrics": {}}),
        encoding="utf-8",
    )
    (metrics_path.parent / "per_case.json").write_text(
        json.dumps(rows),
        encoding="utf-8",
    )
    return metrics_path


def test_multiseed_gate_requires_every_seed_to_pass(tmp_path: Path) -> None:
    runs = [
        (23, write_run(tmp_path, 23)),
        (29, write_run(tmp_path, 29)),
        (31, write_run(tmp_path, 31, failures=30)),
    ]
    report = build_multiseed_report(runs, bootstrap_iterations=20)
    assert report["status"] == "fail"
    assert report["seeds_passing_all_hard_gates"] == 2
    assert report["seeds"][2]["failed_cases"]["room_target_recall"]


def test_multiseed_gate_requires_paired_case_identities(tmp_path: Path) -> None:
    first = write_run(tmp_path, 23)
    second = write_run(tmp_path, 29)
    rows = json.loads((second.parent / "per_case.json").read_text())
    rows[0]["name"] = "different-case"
    (second.parent / "per_case.json").write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="case identities differ"):
        build_multiseed_report(
            [(23, first), (29, second), (31, write_run(tmp_path, 31))],
            bootstrap_iterations=20,
        )
