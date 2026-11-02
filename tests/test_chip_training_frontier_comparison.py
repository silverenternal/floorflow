import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts/build_chip_training_frontier_comparison.py"
    spec = importlib.util.spec_from_file_location("frontier_compare", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_training_frontier_comparison_uses_case_pairs() -> None:
    module = _module()
    baseline = [
        {
            "index": 1,
            "legalized_hard_cost": 1.0,
            "legalized_v_rel": 0.1,
            "legalized_grouping": 1.0,
            "legalized_b2b_hpwl_gap": 0.1,
            "legalized_p2b_hpwl_gap": 0.1,
            "legalized_area_gap": 0.1,
            "legalized_rho_mae": 0.1,
        }
    ]
    candidate = [
        {
            "index": 1,
            "legalized_hard_cost": 0.9,
            "legalized_v_rel": 0.09,
            "legalized_grouping": 1.0,
            "legalized_b2b_hpwl_gap": 0.1,
            "legalized_p2b_hpwl_gap": 0.1,
            "legalized_area_gap": 0.1,
            "legalized_rho_mae": 0.1,
        }
    ]
    paired = module.comparison_rows(
        baseline,
        candidate,
        iterations=10,
        seed=1,
    )
    assert paired["legalized_hard_cost"]["mean_delta"] < 0.0
    assert paired["legalized_v_rel"]["mean_delta"] < 0.0
