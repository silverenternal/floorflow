from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_chip_route_transfer_provenance.py"


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_route_provenance_is_fail_closed_on_collapsed_fresh_pairs(
    tmp_path: Path,
) -> None:
    calibration = {
        "cases": ["nangate45/case"],
        "comparisons": [{"case": "nangate45/case", "seed": 19}],
        "runs": [
            {
                "case": "nangate45/case",
                "seed": 19,
                "arm": arm,
                "macro_placement_tcl_sha256": text,
                "flow": {
                    "command": [
                        "docker",
                        "run",
                        "image",
                        "make",
                        "OR_SEED=1",
                        "GRT_SEED=1",
                        "finish",
                    ]
                },
            }
            for arm, text in (("candidate", "a"), ("control", "b"))
        ],
    }
    observed_v45 = {"pairs": [{"case": "nangate45/case"}]}
    pairs = {
        "status": "reject_collapsed_candidate_control_pairs",
        "collapsed_cases": ["fresh"],
        "rows": [
            {
                "case": "fresh",
                "candidate": {"tcl_sha256": "same"},
                "control": {"tcl_sha256": "same"},
                "checks": {"distinct_tcl": False},
            }
        ],
    }
    verified = {
        "designs": [{"id": "fresh", "flow_config": "missing/config.mk"}]
    }
    inventory = {
        "inventory": [
            {
                "case": "nangate45/case",
                "config": "missing/calibration.mk",
            }
        ]
    }
    candidates = {"candidates": []}
    lock = {
        "image": {
            "local_tag": "image",
            "linux_amd64_manifest_digest": "sha256:manifest",
            "linux_amd64_config_digest": "sha256:config",
        }
    }
    paths = {}
    for name, payload in (
        ("calibration", calibration),
        ("v45", observed_v45),
        ("pairs", pairs),
        ("verified", verified),
        ("inventory", inventory),
        ("candidates", candidates),
        ("lock", lock),
    ):
        paths[name] = tmp_path / f"{name}.json"
        _write(paths[name], payload)
    output = tmp_path / "provenance.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--calibration",
            str(paths["calibration"]),
            "--observed-v45",
            str(paths["v45"]),
            "--chipbench-pairs",
            str(paths["pairs"]),
            "--chipbench-verified",
            str(paths["verified"]),
            "--orfs-inventory",
            str(paths["inventory"]),
            "--orfs-candidates",
            str(paths["candidates"]),
            "--lock",
            str(paths["lock"]),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert report["status"] == "blocked_no_unopened_routed_confirmation"
    assert report["calibration"]["route_recipe_complete_pairs"] == 1
    assert report["opened_chipbench"]["distinct_legalized_pairs"] == 0
    assert report["checks"]["chipbench_pairs_are_distinct"] is False
    assert report["checks"]["at_least_15_fresh_routed_pairs"] is False
