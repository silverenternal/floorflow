#!/usr/bin/env python3
"""Stage ChiPBench-D files in the layout expected by their ORFS configs."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "outputs/reports/chip_chipbench_holdout_manifest_v1_verified.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs/chipbench_holdout_ppa_v1/orfs_designs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chipbench_holdout_orfs_manifest_v1.json",
    )
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def replace_link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    target.symlink_to(os.path.relpath(source, target.parent))


def make_var(path: Path, name: str, default: str) -> str:
    match = re.search(
        rf"^\s*export\s+{name}\s*=\s*(\S+)",
        path.read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    return match.group(1) if match else default


def stage_design(row: dict, output_root: Path) -> Path:
    source_config = ROOT / row["flow_config"]
    parts = Path(row["flow_config"]).parts
    index = parts.index("designs")
    platform = parts[index + 1]
    nickname = row["id"]
    design_dir = output_root / platform / nickname
    if design_dir.exists():
        shutil.rmtree(design_dir)
    design_dir.mkdir(parents=True)
    shutil.copy2(source_config, design_dir / "config.mk")
    verilog = ROOT / row["verilog"]
    replace_link(verilog, design_dir / "rtl" / f"{nickname}{verilog.suffix}")
    replace_link(ROOT / row["sdc"], design_dir / "constraint.sdc")
    for path in row["macro_lefs"]:
        source = ROOT / path
        replace_link(source, design_dir / "lef" / source.name)
    for path in row["libs"]:
        source = ROOT / path
        replace_link(source, design_dir / "lib" / source.name)
    return design_dir


def main() -> None:
    args = parse_args()
    args.manifest = args.manifest.resolve()
    args.output_root = args.output_root.resolve()
    args.output = args.output.resolve()
    manifest = load(args.manifest)
    inventory = []
    for row in manifest["designs"]:
        config = ROOT / row["flow_config"]
        design_name = make_var(config, "DESIGN_NAME", row["id"])
        nickname = make_var(
            config,
            "DESIGN_NICKNAME",
            row["id"].split("/")[-1],
        )
        if row["source"] == "ChiPBench-D":
            design_dir = stage_design(row, args.output_root.resolve())
            config = design_dir / "config.mk"
            platform = design_dir.parent.name
        else:
            platform = row["platform"]
        inventory.append(
            {
                "case": row["id"],
                "design_key": row["id"],
                "config": str(config.relative_to(ROOT)),
                "platform": platform,
                "design_name": design_name,
                "design_nickname": nickname,
            }
        )
    report = {
        "version": "chipbench-holdout-orfs-manifest-v1",
        "status": "staged_before_routed_ppa",
        "inventory": inventory,
        "selection": {
            "selected_cases": [row["id"] for row in manifest["designs"]],
            "independent_designs": len(inventory),
        },
        "source": {
            "verified_manifest": str(args.manifest.relative_to(ROOT)),
            "staged_design_root": str(args.output_root.relative_to(ROOT)),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "designs": len(inventory),
                "staged": sum(
                    row["source"] == "ChiPBench-D"
                    for row in manifest["designs"]
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
