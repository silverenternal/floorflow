from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_chip_v45_collapse_ppa",
    ROOT / "scripts/run_chip_v45_collapse_ppa.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_flow_command_preserves_case_and_arm() -> None:
    args = type("Args", (), {"num_cores": 8, "timeout_seconds": 60})()

    command = MODULE.flow_command(
        args,
        "sky130hd/microwatt",
        "candidate",
        ROOT / "outputs/chip_ppa/placements_v45_collapse_sensitivity/candidate.tcl",
    )

    assert command[command.index("--case") + 1] == "sky130hd/microwatt"
    assert (
        command[command.index("--variant") + 1]
        == "ppa_v45_collapse_sensitivity_candidate_s19"
    )
    assert command[command.index("--flow-target") + 1] == "finish"


def test_collapse_ppa_aggregation_reports_relative_wirelength() -> None:
    result = MODULE.aggregate_pair(
        "p/design",
        {
            "status": "pass",
            "metrics": {
                "detailedroute__route__wirelength": 90.0,
                "finish__power__total": 1.0,
            },
        },
        {
            "status": "pass",
            "metrics": {
                "detailedroute__route__wirelength": 100.0,
                "finish__power__total": 1.0,
            },
        },
    )

    assert result["status"] == "completed"
    assert result["routed_wirelength"]["relative"] == pytest.approx(-0.1)
