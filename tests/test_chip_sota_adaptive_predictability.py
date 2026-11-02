import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts/analyze_chip_adaptive_predictability.py"
    spec = importlib.util.spec_from_file_location("adaptive_predictability", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _case(index: int, costs: list[float], totals: list[float]) -> list[dict]:
    return [
        {
            "index": index,
            "refiner_steps": steps,
            "observable": {"total": total},
            "legalized": {
                "blocks": 40,
                "legalized_hard_cost": cost,
                "legalized_v_rel": cost / 100.0,
                "legalized_feasible": True,
            },
        }
        for steps, cost, total in zip(
            (400, 800, 1600, 3200),
            costs,
            totals,
            strict=True,
        )
    ]


def test_rank_policy_uses_observable_extremum() -> None:
    module = _module()
    cases = {
        1: _case(1, [1.3, 1.2, 1.1, 1.15], [10.0, 9.0, 8.0, 7.0]),
    }
    selected = module.rank_policy_summary(cases, "total", "min")

    assert selected["mean_refiner_steps"] == 3200
    assert abs(selected["legalized_hard_cost"] - 1.15) < 1.0e-12


def test_within_case_rank_consistency_skips_constant_features() -> None:
    module = _module()
    cases = {
        1: _case(1, [1.3, 1.2, 1.1, 1.15], [1.0, 1.0, 1.0, 1.0]),
    }
    result = module.within_case_rank_consistency(cases, "total")

    assert result == {
        "cases": 0,
        "mean_spearman": None,
        "median_spearman": None,
        "positive_fraction": None,
    }
