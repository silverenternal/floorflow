from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/raster2seq_hybrid_tail_repair_v1.json"


def test_tail_screen_used_a_fixed_tail_and_disjoint_controls() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    screen = config["mechanism_screen"]
    tail = set(screen["tail_images"])
    controls = set(screen["control_images"])
    assert not tail & controls
    assert {4, 54, 107}.issubset(tail)
    assert screen["tail_passes_before"] == 0
    assert screen["tail_passes_after"] > screen["tail_passes_before"]
    assert screen["control_regressions"] == 0


def test_tail_repair_keeps_official_metrics_and_room_geometry_fixed() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    candidate = config["candidate"]
    gate = config["full_gate"]
    assert candidate["repair_open_endpoints"]
    assert candidate["rooms_are_not_changed"]
    assert gate["official_raster2seq_metrics_must_be_unchanged"]
    assert gate["wall_area_fraction_must_not_increase"]
    assert config["test503_open"] is False


def test_full_followup_cleared_the_frozen_gate() -> None:
    result = json.loads(CONFIG.read_text(encoding="utf-8"))["full_result"]
    assert result["status"] == "pass"
    assert result["plan_failures_after"] < result["plan_failures_before"]
    assert result["plan_plausibility_after"] >= 0.95
    assert result["wall_continuity_case_pass_after"] >= 0.95
    assert result["functional_watertight"] >= 0.95
    assert result["validity"] == 1.0
    assert result["official_casewise_metric_changes"] == 0
    assert result["wall_area_fraction_casewise_changes"] == 0
