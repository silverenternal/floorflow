import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts/benchmark_chip_capacity_budget.py"
    spec = importlib.util.spec_from_file_location("capacity_budget", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_capacity_config_scales_width_and_heads() -> None:
    module = _module()
    config = module.model_config({"depth": 10, "typed_layers": 1}, 384)
    assert config["hidden"] == 384
    assert config["heads"] == 6
    assert config["depth"] == 10
