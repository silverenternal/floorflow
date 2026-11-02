#!/usr/bin/env python3
"""Statically scan pinned ORFS configs for open-platform macro candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_chip_external_benchmarks import (
    embedded_license_evidence,
    licenses_for_sources,
    macro_masters,
    make_paths,
    parse_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--orfs-root",
        type=Path,
        default=ROOT / "third_party/OpenROAD-flow-scripts",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_orfs_macro_candidate_scan_v1.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    designs_root = args.orfs_root / "flow/designs"
    platforms_root = args.orfs_root / "flow/platforms"
    candidates = []
    no_macro = []
    for config_path in sorted(designs_root.glob("*/*/config.mk")):
        platform = config_path.parents[1].name
        design = config_path.parent.name
        variables = parse_config(config_path, args.orfs_root)
        lef_values = [
            variables.get("ADDITIONAL_LEFS", ""),
            variables.get("WRAP_LEFS", ""),
        ]
        lef_paths = []
        for value in lef_values:
            lef_paths.extend(make_paths(value))
        masters = macro_masters(lef_paths)
        if not masters:
            no_macro.append(f"{platform}/{design}")
            continue
        source_values = []
        for name in (
            "VERILOG_FILES",
            "SYNTH_HDL_FILES",
            "SYNTH_READ_FILES",
            "SDC_FILE",
        ):
            source_values.extend(make_paths(variables.get(name, "")))
        licenses = licenses_for_sources(args.orfs_root, source_values)
        evidence = embedded_license_evidence(source_values)
        candidates.append(
            {
                "id": f"{platform}/{design}",
                "platform": platform,
                "design": design,
                "platform_available": (platforms_root / platform).is_dir(),
                "config": str(config_path.relative_to(ROOT)),
                "macro_master_count": len(masters),
                "macro_masters": sorted(masters),
                "lef_paths": [str(path.relative_to(ROOT)) for path in lef_paths if path.is_file()],
                "macro_placement_tcl": variables.get("MACRO_PLACEMENT_TCL"),
                "rtlmp_max_macro": variables.get("RTLMP_MAX_MACRO"),
                "rtlmp_min_macro": variables.get("RTLMP_MIN_MACRO"),
                "license_files": [str(path.relative_to(ROOT)) for path in licenses],
                "license_evidence": evidence,
                "instantiated_macro_count": None,
                "status": "static_candidate_macro_count_unmeasured",
            }
        )
    available = [
        row for row in candidates if row["platform_available"]
    ]
    output = {
        "version": "chip-orfs-macro-candidate-scan-v1",
        "status": "static_only_macro_counts_and_routes_pending",
        "orfs_root": str(args.orfs_root),
        "configs_scanned": len(candidates) + len(no_macro),
        "static_macro_configs": len(candidates),
        "available_platform_macro_configs": len(available),
        "candidates": candidates,
        "no_macro_configs": no_macro,
        "gate": (
            "Only rows with a measured nonzero instantiated macro count and a "
            "successful pinned-platform route may enter the external cohort."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "configs_scanned": output["configs_scanned"],
                "static_macro_configs": output["static_macro_configs"],
                "available_platform_macro_configs": output[
                    "available_platform_macro_configs"
                ],
                "available_ids": [row["id"] for row in available],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
