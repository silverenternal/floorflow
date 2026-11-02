from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts/run_chip_post_full50k_queue.py"
    spec = importlib.util.spec_from_file_location("post_full50k", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_post_full50k_queue_commands_are_serial_and_predeclared() -> None:
    module = _module()
    args = argparse.Namespace(
        checkpoint=ROOT / "final.pt",
        m10_output=ROOT / "m10.json",
        m6_output=ROOT / "m6.json",
    )
    m10 = module.m10_command(args)
    m6 = module.m6_command(args)
    assert "run_chip_m10_projection_flow_screen.py" in " ".join(m10)
    assert m10[m10.index("--stage") + 1] == "quick"
    assert m10[m10.index("--device") + 1] == "cuda:0"
    assert "run_chip_v46_response_repair.py" in " ".join(m6)
    assert "--wait-pid" not in m6


def test_post_full50k_queue_does_not_launch_from_stale_state_alone() -> None:
    module = _module()
    assert module.process_running(999_999_999) is False


def test_post_full50k_queue_uses_trained_checkpoint_only_after_pass(
    tmp_path: Path,
) -> None:
    module = _module()
    trained = tmp_path / "trained.pt"
    fallback = tmp_path / "fallback.pt"
    report = tmp_path / "report.json"
    trained.write_bytes(b"trained")
    fallback.write_bytes(b"fallback")
    report.write_text('{"status": "complete"}', encoding="utf-8")
    selected, reason = module.select_m10_checkpoint(report, trained, fallback)
    assert selected == trained
    assert reason == "full50k_passed_advance_gate"


def test_post_full50k_queue_falls_back_after_failed_gate(tmp_path: Path) -> None:
    module = _module()
    trained = tmp_path / "trained.pt"
    fallback = tmp_path / "fallback.pt"
    report = tmp_path / "report.json"
    trained.write_bytes(b"trained")
    fallback.write_bytes(b"fallback")
    report.write_text(
        json.dumps({"status": "stopped_at_failed_advance_gate"}),
        encoding="utf-8",
    )
    selected, reason = module.select_m10_checkpoint(report, trained, fallback)
    assert selected == fallback
    assert reason == "full50k_failed_advance_gate_use_frozen_reference"


def test_post_full50k_queue_blocks_unknown_training_status(tmp_path: Path) -> None:
    module = _module()
    report = tmp_path / "report.json"
    report.write_text('{"status": "running"}', encoding="utf-8")
    with pytest.raises(ValueError, match="unresolved full-50k status"):
        module.select_m10_checkpoint(
            report,
            tmp_path / "trained.pt",
            tmp_path / "fallback.pt",
        )
