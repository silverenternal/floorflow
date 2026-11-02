import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts/analyze_chip_outline_seed_stability.py"
    spec = importlib.util.spec_from_file_location("outline_seed_stability", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_outline_seed_aggregation_requires_both_feasible_for_hpwl() -> None:
    module = _module()
    report = {
        "cases": [
            {
                "id": "gsrc/hard/a",
                "candidate": {"selected": {"feasible": True}},
                "control": {"selected": {"feasible": True}},
                "hpwl_relative": -0.1,
            },
            {
                "id": "gsrc/hard/b",
                "candidate": {"selected": {"feasible": True}},
                "control": {"selected": {"feasible": False}},
                "hpwl_relative": 9.0,
            },
        ]
    }
    result = module.analyze([(19, report), (20, report)])
    assert result["candidate_feasible"] == 4
    assert result["control_feasible"] == 2
    assert result["paired_hpwl_relative_mean"] == -0.1
    assert len(result["paired_feasible"]) == 2
