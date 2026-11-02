from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_chip_external_ablation",
    ROOT / "scripts/run_chip_external_ablation.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_completed_full_run_is_locked_to_config_seed(tmp_path: Path) -> None:
    report = tmp_path / "ppa.json"
    report.write_text(
        json.dumps(
            {
                "runs": [
                    {"case": "nangate45/example", "arm": "candidate", "seed": 20},
                    {"case": "nangate45/example", "arm": "candidate", "seed": 19},
                ]
            }
        ),
        encoding="utf-8",
    )
    args = type(
        "Args",
        (),
        {"ppa": report, "config_data": {"seed": 19}},
    )()
    selected = MODULE.get_completed_run(
        args,
        lock={},
        case={"design_key": "nangate45/example"},
        arm="candidate",
    )
    assert selected["seed"] == 19
