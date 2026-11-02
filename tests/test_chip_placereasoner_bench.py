from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from floorflow.chip_flow.placereasoner_bench import (
    benchmark_tasks,
    load_benchmark_contract,
    normalize_coordinates,
    validate_benchmark_root,
    validate_coordinate_count,
)


def _contract(path: Path) -> dict:
    payload = {
        "unique_designs": 1,
        "placement_tasks": 2,
        "repo_commit": "test-commit",
        "dataset_id": "test-dataset",
        "designs": [
            {
                "name": "tiny",
                "top": "tiny_top",
                "macros": 2,
                "clusters": 1,
                "macro_types": 1,
                "clock_ns": 1.0,
            }
        ],
        "aspect_tasks": [
            {"id": "ar1", "def": "def/ar1.def"},
            {"id": "ar2", "def": "def/ar2.def"},
        ],
    }
    path.write_text(json.dumps(payload))
    return payload


def _benchmark(root: Path) -> None:
    design = root / "tiny"
    (design / "def").mkdir(parents=True)
    (design / "lef").mkdir()
    (design / "lib").mkdir()
    (design / "tiny.v").write_text("module tiny_top; endmodule\n")
    (design / "constraint.sdc").write_text("create_clock -period 1 clk\n")
    (design / "def/ar1.def").write_text("VERSION 5.8 ;\n")
    (design / "def/ar2.def").write_text("VERSION 5.8 ;\n")


def test_benchmark_contract_expands_design_aspect_tasks(tmp_path: Path) -> None:
    contract_path = tmp_path / "contract.json"
    _contract(contract_path)
    contract = load_benchmark_contract(contract_path)
    tasks = benchmark_tasks(contract)
    assert [task.task_id for task in tasks] == ["tiny/ar1", "tiny/ar2"]


def test_coordinate_contract_rejects_duplicates_and_wrong_count() -> None:
    coordinates = normalize_coordinates(
        [{"name": "a", "x_um": 1.0, "y_um": 2.0}]
    )
    assert coordinates == [{"name": "a", "x": 1.0, "y": 2.0, "orient": "R0"}]
    with pytest.raises(ValueError, match="duplicate"):
        normalize_coordinates(
            [
                {"name": "a", "x": 1, "y": 2},
                {"name": "a", "x": 3, "y": 4},
            ]
        )
    with pytest.raises(ValueError, match="does not match"):
        validate_coordinate_count(coordinates, 2)


def test_adapter_dry_run_validates_layout_without_openroad(tmp_path: Path) -> None:
    contract_path = tmp_path / "contract.json"
    _contract(contract_path)
    benchmark = tmp_path / "benchmark"
    _benchmark(benchmark)
    tasks = benchmark_tasks(load_benchmark_contract(contract_path))
    assert validate_benchmark_root(benchmark, tasks) == []
    coords = tmp_path / "coords.json"
    coords.write_text(
        json.dumps(
            [
                {"name": "macro_a", "x": 1, "y": 2},
                {"name": "macro_b", "x": 3, "y": 4},
            ]
        )
    )
    report = tmp_path / "report.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/run_chip_placereasoner_external.py",
            "--contract",
            str(contract_path),
            "--benchmark-root",
            str(benchmark),
            "--task",
            "tiny/ar1",
            "--coords",
            str(coords),
            "--dry-run",
            "--report",
            str(report),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    payload = json.loads(report.read_text())
    assert payload["status"] == "dry_run"
    assert payload["macro_count"] == 2


def test_external_place_proc_uses_restart_safe_openroaddb_calls() -> None:
    path = Path("scripts/run_chip_placereasoner_external.py").resolve()
    spec = importlib.util.spec_from_file_location("placereasoner_adapter", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert "setLocation $xdbu $ydbu" in module.PLANB_SAFE_PLACE_PROC
    assert "setOrient $orient" in module.PLANB_SAFE_PLACE_PROC
    assert "setPlacementStatus \"PLACED\"" in module.PLANB_SAFE_PLACE_PROC
