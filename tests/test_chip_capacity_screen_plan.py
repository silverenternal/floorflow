import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts/build_chip_capacity_screen_plan.py"
    spec = importlib.util.spec_from_file_location("capacity_screen_plan", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_equal_flop_steps_respects_measured_ratio() -> None:
    assert _module().equal_flop_steps(20_000, 1.6) == 12_500


def test_generated_capacity_plan_has_both_budget_classes() -> None:
    plan = __import__("json").loads(
        (ROOT / "configs/chip_sota_capacity_screen_v1.json").read_text()
    )
    arms = {row["id"]: row for row in plan["confirmation"]["arms"]}
    assert arms["d256-equal-exposure"]["steps"] == 20_000
    assert arms["d256-equal-flops"]["steps"] < 20_000
    assert arms["d384-equal-flops"]["steps"] < arms["d256-equal-flops"]["steps"]
