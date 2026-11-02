import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts/evaluate_chip_bookshelf_external.py"
    spec = importlib.util.spec_from_file_location("bookshelf_external_eval", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bookshelf_summary_separates_feasible_hpwl() -> None:
    module = _module()
    rows = [
        {
            "candidate": {
                "selected": {
                    "feasible": True,
                    "hpwl": 10.0,
                    "dead_space": 0.1,
                }
            }
        },
        {
            "candidate": {
                "selected": {
                    "feasible": False,
                    "hpwl": 1000.0,
                    "dead_space": 0.9,
                }
            }
        },
    ]

    result = module.summary(rows, "candidate")
    assert result["feasible_fraction"] == 0.5
    assert result["mean_hpwl_all"] == 505.0
    assert result["mean_hpwl_feasible"] == 10.0
