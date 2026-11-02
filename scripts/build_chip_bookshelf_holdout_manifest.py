#!/usr/bin/env python3
"""Freeze the downloaded GSRC/MCNC external structural holdout."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TRINITY = ROOT / "third_party/Trinity/src"
sys.path.insert(0, str(TRINITY))

from trinity.floorplan.data import with_fixed_outline
from trinity.floorplan.data.gsrc import gsrc_case_paths, load_gsrc_case
from trinity.floorplan.data.mcnc import load_mcnc_case, mcnc_case_paths
from trinity.floorplan.data.paths import (
    GSRC_BASE_URL,
    GSRC_SIZES,
    MCNC_BASE_URL,
    MCNC_NAMES,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def source_row(
    suite: str,
    variant: str,
    name: str,
    paths: tuple[Path, Path, Path],
    instance: Any,
) -> dict[str, Any]:
    outlined = with_fixed_outline(instance, gamma=0.10, aspect=1.0, map_pins=False)
    base_url = GSRC_BASE_URL if suite == "gsrc" else MCNC_BASE_URL
    return {
        "id": f"{suite}/{variant.lower()}/{name}",
        "suite": suite,
        "variant": variant,
        "name": name,
        "source_family": f"bookshelf_{suite}_{variant.lower()}",
        "files": [
            {
                "path": relative(path),
                "url": f"{base_url}/{variant}/{path.name}",
                "sha256": digest(path),
                "bytes": path.stat().st_size,
            }
            for path in paths
        ],
        "block_count": instance.block_count,
        "b2b_edges": len(instance.b2b),
        "p2b_edges": len(instance.p2b),
        "terminal_count": len(instance.pins_pos),
        "total_block_area": float(instance.area_targets.sum()),
        "has_reference_placement": instance.gt_positions is not None,
        "aspect_bounds_present": instance.aspect_bounds is not None,
        "fixed_outline": {
            "gamma": 0.10,
            "aspect": 1.0,
            "width": outlined.outline[0],
            "height": outlined.outline[1],
            "map_pins": False,
        },
        "license_status": "unverified_local_research_use_only",
    }


def main() -> None:
    data_root = ROOT / ".data"
    cases = []
    for variant in ("HARD", "SOFT"):
        for size in GSRC_SIZES:
            name = f"n{size}"
            paths = gsrc_case_paths(data_root / "gsrc", size, variant)
            instance = load_gsrc_case(
                size,
                variant,
                root=data_root,
                allow_download=False,
            )
            cases.append(source_row("gsrc", variant, name, paths, instance))
        for name in MCNC_NAMES:
            paths = mcnc_case_paths(data_root / "mcnc", name, variant)
            instance = load_mcnc_case(
                name,
                variant,
                root=data_root,
                allow_download=False,
            )
            cases.append(source_row("mcnc", variant, name, paths, instance))
    ids = [case["id"] for case in cases]
    checks = {
        "at_least_15_cases": len(cases) >= 15,
        "unique_ids": len(ids) == len(set(ids)),
        "all_files_present": all(
            (ROOT / file["path"]).is_file()
            for case in cases
            for file in case["files"]
        ),
        "hard_and_soft_present": {
            (case["suite"], case["variant"]) for case in cases
        }
        == {
            ("gsrc", "HARD"),
            ("gsrc", "SOFT"),
            ("mcnc", "HARD"),
            ("mcnc", "SOFT"),
        },
    }
    report = {
        "version": "chip-v46-bookshelf-holdout-v1",
        "status": "pass" if all(checks.values()) else "fail",
        "scope": (
            "External structural generalization only. These bookshelf cases "
            "are not ORFS routed-PPA evidence."
        ),
        "data_root": ".data",
        "case_count": len(cases),
        "suite_counts": {
            "gsrc": sum(case["suite"] == "gsrc" for case in cases),
            "mcnc": sum(case["suite"] == "mcnc" for case in cases),
        },
        "freeze_policy": (
            "No case may be removed or replaced after candidate outcomes are "
            "opened. Report hard and soft variants separately."
        ),
        "cases": cases,
        "checks": checks,
    }
    output = ROOT / "outputs/reports/chip_v46_bookshelf_holdout_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "checks": checks}, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
