from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _rows(cases: int, *, offset: float = 0.0) -> list[dict]:
    rows = []
    for index in range(cases):
        rows.append(
            {
                "case": f"case_{index:03d}",
                "room_macro_iou": 0.20 + offset,
                "instance_room_f1": 0.10 + offset,
                "instance_type_accuracy": 0.50 + offset,
                "instance_count_abs_error": 5.0 - offset,
                "room_support_iou": 0.70 + offset,
            }
        )
    return rows


def _write_cases(path: Path, cases: int, *, offset: float) -> None:
    weights = ("0.0", "0.25", "0.5", "0.75", "1.0")
    path.write_text(
        json.dumps(
            {
                "case_rows": {
                    weight: _rows(cases, offset=offset)
                    for weight in weights
                }
            }
        ),
        encoding="utf-8",
    )


def test_five_seed_confirmation_reports_paired_and_clustered_statistics(
    tmp_path,
) -> None:
    parent_cal = tmp_path / "parent_cal.json"
    parent_holdout = tmp_path / "parent_holdout.json"
    _write_cases(parent_cal, 8, offset=0.0)
    _write_cases(parent_holdout, 6, offset=0.0)
    seeds = []
    for seed in (19, 23, 29, 31, 37):
        calibration = tmp_path / f"cal_{seed}.json"
        holdout = tmp_path / f"holdout_{seed}.json"
        _write_cases(calibration, 8, offset=0.02)
        _write_cases(holdout, 6, offset=0.01)
        seeds.append(
            {
                "seed": seed,
                "best_blend": 0.5,
                "calibration_cases": str(calibration),
                "holdout_cases": str(holdout),
            }
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "train_cases": 5122,
                "calibration": {"offset": 0, "cases": 8},
                "disjoint_holdout": {"offset": 8, "cases": 6},
                "seeds": seeds,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "report.json"
    table = tmp_path / "report.md"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/build_room_pagefloor_fiveseed_confirmation_v1.py"),
            "--manifest",
            str(manifest),
            "--parent-cal",
            str(parent_cal),
            "--parent-holdout",
            str(parent_holdout),
            "--single-seed-report",
            str(output),
            "--markdown",
            str(table),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["seeds"] == 5
    assert report["gate"]["passed"]
    assert report["aggregate"]["room_macro_iou"]["improved_seeds"] == 5
    assert report["pooled_holdout"]["instance_count_abs_error"]["mean_delta"] < 0
    assert "Five-Seed Confirmation" in table.read_text(encoding="utf-8")
