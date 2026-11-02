from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_gap_analyzer_preserves_provenance_and_counts_families(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "split": "val",
                "validation_cases": 2,
                "test_opened": False,
                "target_used_at_inference": False,
                "cases": [
                    {
                        "case": "over",
                        "room_macro_iou": 0.3,
                        "instance_instance_f1_at_threshold": 0.2,
                        "instance_predicted_instances": 9.0,
                        "instance_target_instances": 7.0,
                        "instance_count_abs_error": 2.0,
                        "plan_plausible": 1.0,
                    },
                    {
                        "case": "under",
                        "room_macro_iou": 0.4,
                        "instance_instance_f1_at_threshold": 0.4,
                        "instance_predicted_instances": 5.0,
                        "instance_target_instances": 7.0,
                        "instance_count_abs_error": 2.0,
                        "plan_plausible": 0.0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "gap.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/analyze_predicted_control_fidelity_gap_v1.py",
            "--input",
            str(source),
            "--output",
            str(output),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["source_protocol"]["test_opened"] is False
    assert payload["overall"]["over_count_cases"] == 1
    assert payload["overall"]["under_count_cases"] == 1
    assert payload["strata"]["count_delta"]["over_minor"]["cases"] == 1
