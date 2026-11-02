from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts/run_chip_range_post_eval.py"
    spec = importlib.util.spec_from_file_location("range_post_eval", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_range_post_eval_reuses_comparison_and_strata_drivers() -> None:
    module = _module()
    args = argparse.Namespace(
        control_eval=ROOT / "control.json",
        candidate_eval=ROOT / "candidate.json",
        range_report=ROOT / "range.json",
        output_root=ROOT / "range",
    )
    comparison = " ".join(module.comparison_command(args))
    highblock = " ".join(module.highblock_command(args))
    assert "build_chip_training_frontier_comparison.py" in comparison
    assert "21-120 frozen subset" in comparison
    assert "analyze_chip_sota_highblock_ladder.py" in highblock
    assert "--steps 20000" in highblock


def test_range_post_eval_detects_missing_pid() -> None:
    module = _module()
    assert module.process_running(999_999_999) is False
