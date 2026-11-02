from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts/build_chip_m64_confirmation_report.py"
    spec = importlib.util.spec_from_file_location("chip_m64_confirmation_report", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _train(seed: int = 21) -> dict:
    return {
        "model_parameters": 100,
        "train_cases": 5000,
        "execution_budget": {
            "planned_steps": 5000,
            "effective_batch": 128,
        },
        "config": {"seed": seed, "data_seed": seed},
        "final_dev": {"total": 0.1},
    }


def _metric(**values: float) -> dict:
    result = {
        "legalized_hard_cost": 1.0,
        "legalized_v_rel": 0.05,
        "legalized_area_gap": 0.1,
        "legalized_b2b_hpwl_gap": 0.1,
        "legalized_p2b_hpwl_gap": 0.1,
        "legalized_grouping": 1.0,
        "legalized_boundary": 1.0,
        "legalized_rho_mae": 0.2,
        "legalized_feasible_fraction": 1.0,
        "cases": 2,
    }
    result.update(values)
    return result


def test_budget_checks_reject_seed_mismatch() -> None:
    module = _module()
    checks = module.budget_checks(_train(), _train(seed=22))
    assert checks["same_training_seed"] is False
    assert checks["same_data_seed"] is False


def test_stage_summary_pairs_case_rows(tmp_path: Path) -> None:
    module = _module()
    candidate = tmp_path / "candidate.json"
    control = tmp_path / "control.json"
    candidate_cases = tmp_path / "candidate_cases.json"
    control_cases = tmp_path / "control_cases.json"
    key = "nfe4_refine400"
    candidate.write_text(
        json.dumps(
            {
                "results": {
                    key: _metric(
                        legalized_hard_cost=0.9,
                        legalized_v_rel=0.04,
                    )
                },
                "_path": str(candidate),
            }
        ),
        encoding="utf-8",
    )
    control.write_text(
        json.dumps({"results": {key: _metric()}, "_path": str(control)}),
        encoding="utf-8",
    )
    rows = [
        {
            "index": 0,
            "legalized_hard_cost": 0.8,
            "legalized_v_rel": 0.04,
            "legalized_area_gap": 0.1,
            "legalized_b2b_hpwl_gap": 0.1,
            "legalized_p2b_hpwl_gap": 0.1,
            "legalized_grouping": 1.0,
            "legalized_boundary": 1.0,
            "legalized_rho_mae": 0.2,
        },
        {
            "index": 1,
            "legalized_hard_cost": 1.0,
            "legalized_v_rel": 0.06,
            "legalized_area_gap": 0.1,
            "legalized_b2b_hpwl_gap": 0.1,
            "legalized_p2b_hpwl_gap": 0.1,
            "legalized_grouping": 1.0,
            "legalized_boundary": 1.0,
            "legalized_rho_mae": 0.2,
        },
    ]
    control_rows = [dict(row) for row in rows]
    for row in control_rows:
        row["legalized_hard_cost"] += 0.1
        row["legalized_v_rel"] += 0.01
    candidate_cases.write_text(json.dumps({key: rows}), encoding="utf-8")
    control_cases.write_text(json.dumps({key: control_rows}), encoding="utf-8")

    summary = module.stage_summary(
        module.read_json(candidate),
        module.read_json(control),
        key,
        iterations=50,
        seed=19,
    )

    assert summary["relative"]["legalized_hard_cost"]["relative"] < 0.0
    assert summary["relative"]["legalized_v_rel"]["relative"] < 0.0
    assert summary["paired"]["legalized_hard_cost"]["mean_delta"] < 0.0
    assert summary["feasibility_both_one"] is True
