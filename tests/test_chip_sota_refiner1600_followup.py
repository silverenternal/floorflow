import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts/run_chip_sota_refiner1600_followup.py"
    spec = importlib.util.spec_from_file_location("refiner1600_followup", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_last_arm_uses_only_completed_ladder_rows() -> None:
    module = _module()
    report = {
        "steps": [
            {"steps": 20_000},
            {"steps": 50_000},
        ]
    }
    assert module.last_arm(report)["steps"] == 50_000


def test_last_arm_rejects_empty_ladder() -> None:
    module = _module()
    try:
        module.last_arm({"steps": []})
    except ValueError as error:
        assert "no completed arms" in str(error)
    else:
        raise AssertionError("empty ladder should be rejected")
