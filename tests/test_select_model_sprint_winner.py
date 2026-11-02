from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load_main():
    path = Path(__file__).resolve().parents[1] / "scripts/select_model_sprint_winner.py"
    spec = importlib.util.spec_from_file_location("select_model_sprint_winner", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.main


def _write(path: Path, quality: float, coverage: float) -> None:
    path.write_text(
        json.dumps(
            {
                "checkpoint": f"checkpoint-{quality}",
                "aggregate": {
                    "quality_mean": quality,
                    "coverage_worst": coverage,
                },
            }
        ),
        encoding="utf-8",
    )


def test_winner_selection_enforces_relative_gain(tmp_path: Path, monkeypatch) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    output = tmp_path / "winner.json"
    _write(baseline, 0.10, 0.8)
    _write(candidate, 0.11, 0.75)
    monkeypatch.setattr(
        "sys.argv",
        [
            "select",
            "--baseline",
            str(baseline),
            "--reports",
            str(candidate),
            "--minimum-relative-gain",
            "0.15",
            "--output",
            str(output),
        ],
    )

    _load_main()()

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "no_candidate_above_gate"
    assert report["winner"]["relative_gain"] == pytest.approx(0.1)
    assert report["best_of_n_used"] is False
