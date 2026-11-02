from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_main():
    path = Path(__file__).resolve().parents[1] / "scripts/analyze_user_study_responses.py"
    spec = importlib.util.spec_from_file_location("user_study_analysis", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.main


def test_user_study_analysis_reports_condition_rates(tmp_path: Path, monkeypatch) -> None:
    answer_key = {
        "cases": [
            {
                "case_id": "case",
                "requested_style_id": 0,
                "slot_conditions": {
                    "A": "style_conditioned",
                    "B": "no_style",
                    "C": "retrieval",
                },
            }
        ]
    }
    response = {
        "participant": {
            "role": "professional",
            "consent": True,
            "attention_ok": True,
        },
        "cases": [
            {
                "case_id": "case",
                "preferred_slot": "A",
                "ratings": {
                    "A": {
                        "style_fit_1_to_7": 7,
                        "legality_pass_fail": "pass",
                        "confidence_1_to_7": 6,
                    },
                    "B": {
                        "style_fit_1_to_7": 3,
                        "legality_pass_fail": "pass",
                        "confidence_1_to_7": 4,
                    },
                    "C": {
                        "style_fit_1_to_7": 4,
                        "legality_pass_fail": "fail",
                        "confidence_1_to_7": 3,
                    },
                },
            }
        ],
    }
    key_path = tmp_path / "key.json"
    response_path = tmp_path / "response.json"
    output = tmp_path / "analysis.json"
    key_path.write_text(json.dumps(answer_key), encoding="utf-8")
    response_path.write_text(json.dumps(response), encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "analyze",
            "--responses",
            str(response_path),
            "--answer-key",
            str(key_path),
            "--output",
            str(output),
        ],
    )

    _load_main()()

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "analyzed"
    assert report["conditions"]["style_conditioned"]["preferred_rate"]["mean"] == 1.0
    assert report["conditions"]["style_conditioned"]["style_fit_mean"] == 7.0
    assert report["conditions"]["retrieval"]["legality_pass_rate"] == 0.0
