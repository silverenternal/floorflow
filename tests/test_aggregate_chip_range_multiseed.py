from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts/aggregate_chip_range_multiseed.py"
    spec = importlib.util.spec_from_file_location("range_aggregate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_range_aggregate_uses_seed19_and_multiseed_paths() -> None:
    module = _module()
    seed19 = module.seed_paths(19)
    seed20 = module.seed_paths(20)
    assert seed19[0] == ROOT / "outputs/reports/chip_range_control_comparison_v1.json"
    assert "seed20/control" in str(seed20[1])
    assert "seed20/range" in str(seed20[2])


def test_range_aggregate_supports_refiner1600_paths() -> None:
    module = _module()
    seed19 = module.seed_paths(19, "ref1600")
    seed21 = module.seed_paths(21, "ref1600")
    assert seed19[0].name == "chip_range_comparison_seed19_refine1600.json"
    assert seed19[1].name == "eval_20000_refine1600_cases.json"
    assert seed21[0].name == "chip_range_comparison_seed21_refine1600.json"
    assert "seed21/range" in str(seed21[2])


def test_range_aggregate_clusters_by_case_index() -> None:
    module = _module()
    rows = module.case_rows(
        ROOT / "outputs/chip_hierarchy/sota_range_control_v1/eval_20000_cases.json",
        "nfe4_refine400",
    )
    assert len(rows) == 32
    assert all("index" in row for row in rows)
