import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts/audit_chip_current_evidence.py"
    spec = importlib.util.spec_from_file_location("current_evidence", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_current_evidence_gate_keeps_cvpr_experiment_unready() -> None:
    gate = __import__("json").loads(
        (ROOT / "outputs/reports/chip_current_evidence_gate_v1.json").read_text()
    )
    assert gate["publishable_cvpr_evidence_ready"] is False
    assert gate["artifact_integrity"]["status"] == "pass"
    assert len(gate["blockers"]) >= 4
    assert gate["scientific_gates"]["fresh_external_supply_sufficient"] is False
    assert gate["fresh_external_supply"]["design_count"] == 15
    assert gate["fresh_external_supply"]["candidate_outcomes_opened"] is False
    assert gate["fresh_external_supply"]["graph_audits_ready"] is True
    assert gate["fresh_external_supply"]["flow_configs_ready"] is True
    assert gate["fresh_external_supply"]["pending_macro_preflight"] == [
        "sky130hd/chameleon"
    ]
    assert (
        gate["scientific_gates"][
            "training_20k_combined_recipe_beats_production_control"
        ]
        is False
    )
    assert gate["training_frontier"]["cost_relative"] > 0
    assert gate["training_frontier"]["subset_cost_relative"] > 0
    assert gate["scientific_gates"]["training_full50k_exposure_pass"] is False
    assert gate["training_full50k"]["status"] == "stopped_at_failed_advance_gate"
    assert (
        gate["scientific_gates"]["training_subset_control_matches_production"]
        is False
    )
    assert gate["scientific_gates"]["u1_quick_mean_pass"] is True
    assert gate["scientific_gates"]["u1_confirmation_complete"] is True
    assert gate["scientific_gates"]["u1_confirmation_pass"] is False
    assert gate["scientific_gates"]["m10_operator_smoke_complete"] is True
    assert gate["scientific_gates"]["m10_frozen_screen_complete"] is True
    assert gate["scientific_gates"]["m10_screen_pass"] is False
    assert gate["m10_projection_flow"]["quick_cases"] == 32
    assert gate["m10_projection_flow"]["confirmation_cases"] == 32
    assert gate["m10_projection_flow"]["status"] == "rejected_confirmation"
    assert gate["m10_projection_flow"]["nfe4_smoke_cost_delta"] > 0
    assert gate["scientific_gates"]["m6_response_repair_complete"] is True
    assert gate["scientific_gates"]["m6_deployable_selector_supported"] is False
    assert gate["m6_response_repair"]["missing_response_count"] == 0
    assert gate["m6_response_repair"]["decision_status"] == (
        "reject_deployable_selector"
    )
