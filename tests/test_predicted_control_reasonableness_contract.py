from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/predicted_control_reasonableness_contract_v1.json"


def _contract() -> dict:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def test_contract_separates_official_adapter_and_teacher_sources() -> None:
    contract = _contract()
    sources = contract["source_revisions"]
    assert sources["raster2seq"]["commit"] == "a6c4e27a68d11d7a459f6e4a2601fd887227dd1a"
    assert sources["fpvec_lab"]["commit"] == "b4b4a169a19b098ed3d1c39f67dfc7f89ed6b566"
    assert sources["housemind"]["official_inference"] is False
    assert contract["target_policy"]["official_and_adapter_metrics_are_not_interchangeable"]
    assert contract["target_policy"]["teacher_predictions_are_controls_not_ground_truth"]
    assert contract["target_policy"]["test503_open"] is False


def test_contract_keeps_noninterchangeable_metrics_separate() -> None:
    contract = _contract()
    metrics = contract["metrics"]
    assert metrics["official_room_iou"]["role"] == "official external reconstruction metric"
    assert "never substituted" in metrics["adapter_room_macro_iou"]["role"]
    assert metrics["raw_watertightness"]["source"] == "fpvec_lab"
    assert metrics["componentwise_watertightness"]["source"] == "floorflow_generation_plausibility"
    pairs = {tuple(pair) for pair in contract["non_interchangeable_pairs"]}
    assert ("official_room_iou", "adapter_room_macro_iou") in pairs
    assert ("raw_watertightness", "componentwise_watertightness") in pairs


def test_current_reasonableness_decision_matches_evidence() -> None:
    decision = _contract()["current_reasonableness_decision"]
    standalone = decision["standalone_floorflow_five_seed_val491"]
    assert standalone["status"] == "pass"
    assert standalone["validity_mean"] == 1.0
    assert standalone["functional_watertight_mean"] >= 0.95
    assert standalone["wall_continuity_mean"] >= 0.95
    assert standalone["plan_plausibility_mean"] >= 0.95

    hybrid = decision["raster2seq_proposal_hybrid_val503_v6"]
    assert hybrid["status"] == "pass"
    assert hybrid["validity"] == 1.0
    assert hybrid["functional_watertight"] >= 0.95
    assert hybrid["wall_continuity_case_pass"] >= 0.95
    assert hybrid["plan_plausibility"] >= 0.95
    assert hybrid["plan_failure_cases"] == 13
