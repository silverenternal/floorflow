from __future__ import annotations

import importlib.util
from pathlib import Path


def _audit():
    path = Path("scripts/audit_chip_architecture.py")
    spec = importlib.util.spec_from_file_location("chip_architecture_audit", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.audit_protocol_actions


def test_protocol_action_map_points_to_existing_model_paths() -> None:
    report, checks = _audit()(
        Path("configs/chip_protocol_actions_v2.json")
    )
    assert report["status"] == "pass", report["errors"]
    assert all(checks.values())


def test_protocol_action_map_keeps_external_stages_external() -> None:
    report, _ = _audit()(
        Path("configs/chip_protocol_actions_v2.json")
    )
    gaps = {gap["id"]: gap for gap in report["active_gap_queue"]}
    assert {
        "validated_globalplace_response",
        "L1_constraint_necessity",
        "routing_feedback_after_response",
        "legalized_pair_identity",
        "hyperedge_data_gap",
        "orientation_data_gap",
        "halo_channel_data_gap",
        "fixed_outline_l0_conditioning",
    }.issubset(gaps)
