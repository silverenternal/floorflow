from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts/run_chip_range_seed_replication.py"
    spec = importlib.util.spec_from_file_location("range_multiseed", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_range_multiseed_clone_preserves_budgets_and_changes_seed() -> None:
    module = _module()
    config = {
        "status": "old",
        "training": {"seed": 19},
        "steps": [1, 2],
        "evaluation": {"device": "cuda:0"},
    }
    cloned = module.clone_config(config, 20, "range_21_120")
    assert cloned["training"]["seed"] == 20
    assert cloned["steps"] == [20000]
    assert cloned["evaluation"]["device"] == "cuda:1"
    assert config["training"]["seed"] == 19


def test_range_multiseed_comparison_is_same_protocol() -> None:
    module = _module()
    command = " ".join(
        module.comparison_command(
            ROOT / "control.json",
            ROOT / "range.json",
            ROOT / "out.json",
            21,
        )
    )
    assert "build_chip_training_frontier_comparison.py" in command
    assert "seed21 dev32" in command
