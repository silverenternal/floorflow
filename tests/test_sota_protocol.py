"""Tests for the canonical residential SOTA protocol."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

from floorflow.evaluation.fevp_metric_contracts import FEVPTolerances
from floorflow.evaluation.sota_protocol import (
    CanonicalFloorplan,
    canonical_fingerprint,
    evaluate_canonical_pair,
    validate_floorplan,
)

ROOT = Path(__file__).resolve().parents[1]
_AUDIT_PATH = ROOT / "scripts/audit_sota_benchmark_protocol.py"
_SPEC = importlib.util.spec_from_file_location("sota_protocol_audit", _AUDIT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_AUDIT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_AUDIT)
sample_plan = _AUDIT.sample_plan
attack_plans = _AUDIT.attack_plans
taxonomy = _AUDIT.taxonomy


def test_config_freezes_cross_method_graph_boundaries() -> None:
    config = json.loads(
        (ROOT / "configs/sota_benchmark_protocol_v2.json").read_text(
            encoding="utf-8"
        )
    )
    graph = config["graph_contract"]
    assert graph["primary_protocol"] == "unique_room_type_adjacency"
    assert graph["secondary_protocol"] == "type_geometry_hungarian"
    assert "Never pool" in graph["pooling_rule"]
    assert config["tracks"]["oracle_control_diagnostic"]["leaderboard"] is False


def test_canonical_output_roundtrips_without_serialization_drift() -> None:
    plan = sample_plan()
    restored = CanonicalFloorplan.from_dict(
        json.loads(json.dumps(plan.to_dict()))
    )
    assert canonical_fingerprint(restored) == canonical_fingerprint(plan)
    assert not validate_floorplan(
        plan,
        taxonomy(),
        tolerances=FEVPTolerances(),
        require_complete_cad=True,
    )


def test_identity_case_has_exact_geometry_and_graph_metrics() -> None:
    plan = sample_plan()
    result = evaluate_canonical_pair(
        plan,
        plan,
        taxonomy(),
        tolerances=FEVPTolerances(),
    )
    assert result["room_iou"] == 1.0
    assert result["node_f1"] == 1.0
    assert result["edge_overlap"] == 1.0
    assert result["ged"] == 0.0
    assert result["validity"] == 1.0
    assert result["conditional_watertightness"] == 1.0


def test_all_deterministic_attacks_change_the_declared_metric() -> None:
    base = sample_plan()
    expected = {
        "remove_room": ("node_f1", "instance_node_f1", "instance_ged"),
        "remove_adjacency": ("edge_overlap", "instance_edge_overlap"),
        "move_opening": ("opening_f1",),
        "remove_furniture": ("furniture_f1",),
        "overlap_rooms": ("validity",),
        "remove_wall": ("wall_f1",),
    }
    clean = evaluate_canonical_pair(
        base,
        base,
        taxonomy(),
        tolerances=FEVPTolerances(),
    )
    for name, attacked in attack_plans(base).items():
        result = evaluate_canonical_pair(
            attacked,
            base,
            taxonomy(),
            tolerances=FEVPTolerances(),
        )
        for metric in expected[name]:
            assert result[metric] != clean[metric], (name, metric)


def test_validator_rejects_illegal_cross_references_and_missing_support() -> None:
    plan = sample_plan()
    broken = replace(
        plan,
        rooms=(replace(plan.rooms[0], support=None), plan.rooms[1]),
        adjacency=(("room_1", "missing"),),
        openings=(replace(plan.openings[0], wall_id="missing"),),
    )
    errors = validate_floorplan(
        broken,
        taxonomy(),
        tolerances=FEVPTolerances(),
        require_complete_cad=True,
    )
    assert any("missing support" in error for error in errors)
    assert any("unknown room" in error for error in errors)
    assert any("unknown wall" in error for error in errors)
