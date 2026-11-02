#!/usr/bin/env python3
"""Audit executability of contemporary boundary-conditioned baselines."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def source_audit(
    name: str,
    repository: Path,
    *,
    data_paths: tuple[Path, ...],
    weight_paths: tuple[Path, ...],
    test_scripts: tuple[Path, ...],
) -> dict:
    commit = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    data = {str(path.relative_to(ROOT)): path.exists() for path in data_paths}
    weights = {
        str(path.relative_to(ROOT)): path.exists() for path in weight_paths
    }
    scripts = {
        str(path.relative_to(ROOT)): path.exists() for path in test_scripts
    }
    return {
        "name": name,
        "repository": str(repository.relative_to(ROOT)),
        "commit": commit,
        "data_ready": bool(data) and all(data.values()),
        "weights_ready": bool(weights) and all(weights.values()),
        "test_entrypoints_ready": bool(scripts) and all(scripts.values()),
        "data_paths": data,
        "weight_paths": weights,
        "test_entrypoints": scripts,
    }


def main() -> None:
    gsdiff_root = ROOT / "third_party/GSDiff"
    gsdiff = source_audit(
        "GSDiff",
        gsdiff_root,
        data_paths=(
            gsdiff_root / "datasets/rplandata/Data/floorplan_dataset",
        ),
        weight_paths=(
            gsdiff_root
            / "outputs/structure-56-36-interval1000/model_stage2_best_065000.pt",
            gsdiff_root / "outputs/structure-81-106-3",
        ),
        test_scripts=(gsdiff_root / "scripts/test_boun.py",),
    )
    remote = {
        "FMLM": {
            "citation": "CVPR 2026; arXiv:2604.04859",
            "project_page": "https://mapooon.github.io/FMLPage",
            "code_release": "not advertised on the project page",
            "local_executable": False,
        },
        "TLC-Plan": {
            "citation": "arXiv:2602.07100",
            "repository": "https://github.com/rosolose/TLC-PLAN",
            "code_release": "repository currently exposes README and teaser asset only",
            "local_executable": False,
        },
    }
    executable = [
        name
        for name, item in (("GSDiff", gsdiff),)
        if all(
            item[key]
            for key in (
                "data_ready",
                "weights_ready",
                "test_entrypoints_ready",
            )
        )
    ]
    report = {
        "status": "complete_audit_baseline_not_runnable",
        "checked_on": "2026-10-26",
        "executable_native_baselines": executable,
        "gsdiff": gsdiff,
        "unavailable_remote_baselines": remote,
        "data_access": {
            "rplan": (
                "Official RPLAN access is gated by the authors' Google Form; "
                "no local RPLAN tree is present."
            )
        },
        "policy": (
            "Do not compare RPLAN-native baseline scores with CubiCasa or "
            "FloorSet scores. A baseline can enter a matched table only after "
            "its exact split, condition interface, checkpoint, and scorer are "
            "reproduced."
        ),
        "next_actions": [
            "Obtain authorized RPLAN access and run GSDiff boundary inference.",
            "Record the decoder/readout mapping before measuring legality.",
            "Keep FMLM and TLC-Plan as related-work references unless code and checkpoints are released.",
        ],
        "best_of_n_used": False,
        "target_derived_control": False,
    }
    output = (
        ROOT
        / "outputs/reports/contemporary_boundary_baseline_audit_v1.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
