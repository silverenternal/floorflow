from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_chipbench_holdout_orfs_manifest.py"


def test_relative_input_paths_are_resolved_from_repo_root() -> None:
    verified = {
        "designs": [
            {
                "id": "case",
                    "platform": "nangate45",
                "flow_config": "third_party/example/config.mk",
            }
        ]
    }
    pairs = {
        "status": "frozen_pair_manifest_ready",
        "all_routes_matched": True,
        "all_legalized_feasible": True,
        "all_zero_overlap": True,
        "all_candidate_control_tcl_distinct": True,
        "rows": [{"case": "case", "checks": {"distinct_tcl": True}}],
    }
    with tempfile.TemporaryDirectory(dir=ROOT) as temp:
        directory = Path(temp)
        verified_path = directory / "verified.json"
        pairs_path = directory / "pairs.json"
        output_path = directory / "orfs.json"
        verified_path.write_text(json.dumps(verified))
        pairs_path.write_text(json.dumps(pairs))
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--verified-manifest",
                str(verified_path.relative_to(ROOT)),
                "--pair-manifest",
                str(pairs_path.relative_to(ROOT)),
                "--output",
                str(output_path.relative_to(ROOT)),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert json.loads(output_path.read_text())["status"] == (
            "staged_before_routed_ppa"
        )


def test_collapsed_pair_manifest_blocks_orfs_staging(
    tmp_path: Path,
) -> None:
    verified = {
        "designs": [
            {
                "id": "case",
                "platform": "nangate45",
                "flow_config": "third_party/example/config.mk",
            }
        ]
    }
    pairs = {
        "status": "reject_collapsed_candidate_control_pairs",
        "all_routes_matched": True,
        "all_legalized_feasible": True,
        "all_zero_overlap": True,
        "all_candidate_control_tcl_distinct": False,
        "rows": [{"case": "case", "checks": {"distinct_tcl": False}}],
    }
    verified_path = tmp_path / "verified.json"
    pairs_path = tmp_path / "pairs.json"
    output_path = tmp_path / "orfs.json"
    verified_path.write_text(json.dumps(verified), encoding="utf-8")
    pairs_path.write_text(json.dumps(pairs), encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--verified-manifest",
            str(verified_path),
            "--pair-manifest",
            str(pairs_path),
            "--output",
            str(output_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 1
    report = json.loads(output_path.read_text())
    assert report["status"] == "blocked_invalid_placement_pairs"
    assert report["inventory"] == []
    assert report["pair_gate"]["all_tcl_distinct"] is False
