import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts/analyze_chip_bookshelf_coverage.py"
    spec = importlib.util.spec_from_file_location("bookshelf_coverage", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_coverage_analysis_keeps_hpwl_to_paired_feasible_cases() -> None:
    module = _module()
    report = {
        "cases": [
            {
                "id": "gsrc/hard/a",
                "candidate": {"selected": {"feasible": True, "hpwl": 10.0}},
                "control": {"selected": {"feasible": True, "hpwl": 11.0}},
            },
            {
                "id": "gsrc/hard/b",
                "candidate": {"selected": {"feasible": True, "hpwl": 999.0}},
                "control": {"selected": {"feasible": False, "hpwl": 1.0}},
            },
        ]
    }
    result = module.analyze(report, iterations=100, seed=19)

    assert result["candidate_feasible"] == 2
    assert result["control_feasible"] == 1
    assert result["both_feasible"] == 1
    assert result["feasible_pair_hpwl"]["cases"] == 1
    assert result["feasible_pair_hpwl"]["mean_delta"] == -1.0
