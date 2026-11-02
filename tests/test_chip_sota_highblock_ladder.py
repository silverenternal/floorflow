import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts/analyze_chip_sota_highblock_ladder.py"
    spec = importlib.util.spec_from_file_location("highblock_ladder_analysis", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_highblock_strata_are_disjoint_and_cover_extremes() -> None:
    module = _module()
    rows = [
        {
            "blocks": blocks,
            "legalized_hard_cost": 1.0,
            "legalized_v_rel": 0.02,
            "legalized_feasible": True,
        }
        for blocks in (21, 40, 60, 61, 90, 91, 100, 120)
    ]
    slices = module.stratify(rows)

    assert slices["low_21_60"]["cases"] == 3
    assert slices["mid_61_90"]["cases"] == 2
    assert slices["high_91_120"]["cases"] == 3
    assert slices["extreme_100_120"]["cases"] == 2


def test_relative_delta_uses_lower_is_better_sign() -> None:
    module = _module()
    previous = {"legalized_hard_cost": 1.0, "legalized_v_rel": 0.02}
    current = {"legalized_hard_cost": 0.98, "legalized_v_rel": 0.021}

    delta = module.relative_delta(previous, current)
    assert abs(delta["legalized_hard_cost"] + 0.02) < 1.0e-12
    assert abs(delta["legalized_v_rel"] - 0.05) < 1.0e-12


def test_case_rows_support_refiner1600_evaluation_template(tmp_path: Path) -> None:
    module = _module()
    evaluation = tmp_path / "eval_50000_refine1600.json"
    cases = tmp_path / "eval_50000_refine1600_cases.json"
    evaluation.write_text("{}")
    cases.write_text(
        json.dumps(
            {
                "nfe4_refine1600": [
                    {
                        "index": 1,
                        "blocks": 60,
                        "legalized_hard_cost": 1.0,
                        "legalized_v_rel": 0.02,
                        "legalized_feasible": True,
                    }
                ]
            }
        )
    )

    rows = module.case_rows(
        tmp_path,
        50_000,
        "nfe4_refine1600",
        "eval_{steps}_refine1600.json",
    )
    assert rows[0]["index"] == 1
