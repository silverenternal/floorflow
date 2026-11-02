from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_chipbench_holdout_pair_manifest.py"


def _write_pair(
    root: Path,
    *,
    case: str,
    candidate_tcl: str,
    control_tcl: str,
) -> tuple[Path, Path]:
    arms = {}
    for arm, content in (
        ("candidate", candidate_tcl),
        ("control", control_tcl),
    ):
        tcl = root / f"{case}_{arm}.tcl"
        boxes = root / f"{case}_{arm}.json"
        report = root / f"{case}_{arm}_placement.json"
        tcl.write_text(content, encoding="utf-8")
        boxes.write_text("{}", encoding="utf-8")
        report.write_text(
            json.dumps(
                {
                    "legalizer_route": "expanded",
                    "geometry": {
                        "legalized_output": {
                            "checker_feasible": True,
                            "overlap_pair_count": 0,
                        }
                    },
                    "outputs": {
                        "tcl": str(tcl),
                        "boxes": str(boxes),
                    },
                }
            ),
            encoding="utf-8",
        )
        arms[arm] = {
            "status": "pass",
            "case": case,
            "placement_report": str(report),
        }
    candidate = root / "candidate.json"
    control = root / "control.json"
    candidate.write_text(json.dumps({"runs": [arms["candidate"]]}))
    control.write_text(json.dumps({"runs": [arms["control"]]}))
    return candidate, control


def _run(tmp_path: Path, candidate: Path, control: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--candidate",
            str(candidate),
            "--control",
            str(control),
            "--skip-forced",
            "--output",
            str(tmp_path / "pairs.json"),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_pair_manifest_rejects_collapsed_tcl(tmp_path: Path) -> None:
    candidate, control = _write_pair(
        tmp_path,
        case="collapsed",
        candidate_tcl="same",
        control_tcl="same",
    )
    result = _run(tmp_path, candidate, control)
    assert result.returncode == 1
    report = json.loads((tmp_path / "pairs.json").read_text())
    assert report["status"] == "reject_collapsed_candidate_control_pairs"
    assert report["collapsed_cases"] == ["collapsed"]


def test_pair_manifest_accepts_distinct_legal_tcl(tmp_path: Path) -> None:
    candidate, control = _write_pair(
        tmp_path,
        case="distinct",
        candidate_tcl="candidate",
        control_tcl="control",
    )
    result = _run(tmp_path, candidate, control)
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "pairs.json").read_text())
    assert report["status"] == "frozen_pair_manifest_ready"
    assert report["all_candidate_control_tcl_distinct"]
