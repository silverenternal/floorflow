from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts/run_chip_m10_projection_flow_screen.py"
    spec = importlib.util.spec_from_file_location("m10_screen", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_m10_config_freeze_has_quick_and_confirmation_stages() -> None:
    config = _module().load_json(
        ROOT / "configs/chip_m10_projection_consistent_flow_v1.json"
    )
    expected = {"post_step", "projection_consistent", "final_only"}
    assert set(config["projection_modes"]) == expected
    for stage in ("quick", "confirmation"):
        assert config["stages"][stage]["refiner_steps"][0] == 0
    assert set(config["stages"]["quick"]["projection_modes"]) == expected
    assert set(config["stages"]["confirmation"]["projection_modes"]) == {
        "post_step",
        "final_only",
    }
    assert config["selection"]["selected_mode"] == "final_only"
    assert config["advance_gate"]["full_data_allowed"] is False
    assert config["source_review"]["license"] == "not_present_on_2026_10_30"


def test_upgrade_registry_prioritizes_m10_and_keeps_u1_rejected() -> None:
    module = _module()
    config = module.load_json(
        ROOT / "configs/chip_upgrade_candidates_v1.json"
    )
    candidates = {row["id"]: row for row in config["candidates"]}
    assert candidates["M10-projection-consistent-flow"]["priority"] == 0
    assert candidates["M10-projection-consistent-flow"]["status"] == (
        "rejected_at_confirmation"
    )
    assert candidates["U1-dynamic-state-graph-relay"]["status"] == (
        "rejected_at_confirmation"
    )
    assert config["blocking_snapshot"]["full50k_status"] == (
        "rejected_same_device_advance_gate"
    )


def test_candidate_registry_keeps_m10_confirmation_negative() -> None:
    module = _module()
    registry = module.load_json(ROOT / "configs/chip_candidate_registry.json")
    rejected = {
        row["id"]: row for row in registry["rejected_screens"]
    }
    row = rejected["m10_final_only_terminal_projection"]
    assert row["status"] == "rejected_confirmation"
    assert row["quick_cost_relative"] < 0.0
    assert row["confirmation_cost_relative"] > 0.0
    assert row["confirmation_v_rel_relative"] > 0.0


def test_m10_comparison_marks_only_both_metric_mean_improvement() -> None:
    module = _module()
    config = {
        "nfes": [1],
        "refiner_steps": [0],
        "projection_modes": ["post_step", "projection_consistent"],
    }
    baseline = [
        {
            "index": 0,
            "legalized_hard_cost": 1.2,
            "legalized_v_rel": 0.04,
            "legalized_area_gap": 0.03,
            "legalized_grouping": 0.5,
            "legalized_boundary": 0.5,
        },
        {
            "index": 1,
            "legalized_hard_cost": 1.0,
            "legalized_v_rel": 0.02,
            "legalized_area_gap": 0.02,
            "legalized_grouping": 0.4,
            "legalized_boundary": 0.3,
        },
    ]
    candidate = [
        {
            "index": 0,
            "legalized_hard_cost": 1.1,
            "legalized_v_rel": 0.035,
            "legalized_area_gap": 0.025,
            "legalized_grouping": 0.45,
            "legalized_boundary": 0.45,
        },
        {
            "index": 1,
            "legalized_hard_cost": 0.95,
            "legalized_v_rel": 0.019,
            "legalized_area_gap": 0.019,
            "legalized_grouping": 0.38,
            "legalized_boundary": 0.29,
        },
    ]
    summaries = {
        "post_step_nfe1_refine0": {
            "legalized_hard_cost": 1.1,
            "legalized_v_rel": 0.03,
            "legalized_feasible_fraction": 1.0,
        },
        "projection_consistent_nfe1_refine0": {
            "legalized_hard_cost": 1.025,
            "legalized_v_rel": 0.027,
            "legalized_feasible_fraction": 1.0,
        },
    }
    rows = {
        "post_step_nfe1_refine0": baseline,
        "projection_consistent_nfe1_refine0": candidate,
    }
    comparisons = module.compare_modes(summaries, rows, "post_step", config)
    result = comparisons["projection_consistent_nfe1_refine0"]
    assert result["mean_pass"] is True
    assert result["paired"]["legalized_hard_cost"]["improved"] == 2
    assert result["paired"]["legalized_v_rel"]["ci_low"] < 0.0


def test_m10_comparison_rejects_one_metric_only() -> None:
    module = _module()
    config = {
        "nfes": [1],
        "refiner_steps": [0],
        "projection_modes": ["post_step", "final_only"],
    }
    baseline = [
        {
            "index": 0,
            "legalized_hard_cost": 1.0,
            "legalized_v_rel": 0.02,
            "legalized_area_gap": 0.02,
            "legalized_grouping": 0.4,
            "legalized_boundary": 0.3,
        }
    ]
    candidate = [
        {
            "index": 0,
            "legalized_hard_cost": 0.9,
            "legalized_v_rel": 0.021,
            "legalized_area_gap": 0.02,
            "legalized_grouping": 0.4,
            "legalized_boundary": 0.3,
        }
    ]
    summaries = {
        "post_step_nfe1_refine0": {
            "legalized_hard_cost": 1.0,
            "legalized_v_rel": 0.02,
            "legalized_feasible_fraction": 1.0,
        },
        "final_only_nfe1_refine0": {
            "legalized_hard_cost": 0.9,
            "legalized_v_rel": 0.021,
            "legalized_feasible_fraction": 1.0,
        },
    }
    rows = {
        "post_step_nfe1_refine0": baseline,
        "final_only_nfe1_refine0": candidate,
    }
    comparisons = module.compare_modes(summaries, rows, "post_step", config)
    assert comparisons["final_only_nfe1_refine0"]["mean_pass"] is False


def test_m10_distribution_summary_handles_empty_strata() -> None:
    module = _module()
    assert module.distribution_summary([]) == {"cases": 0}
    rows = [
        {
            "legalized_hard_cost": 1.0,
            "legalized_v_rel": 0.01,
        },
        {
            "legalized_hard_cost": 1.1,
            "legalized_v_rel": 0.03,
        },
    ]
    summary = module.distribution_summary(rows)
    assert summary["cases"] == 2
    assert summary["cost_p50"] == 1.05
    assert summary["v_rel_p95"] >= summary["v_rel_p50"]
