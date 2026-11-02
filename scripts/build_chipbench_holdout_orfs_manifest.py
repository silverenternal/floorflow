#!/usr/bin/env python3
"""Build an ORFS-compatible inventory for the verified 15-design cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verified-manifest",
        type=Path,
        default=ROOT / "outputs/reports/chip_chipbench_holdout_manifest_v1_verified.json",
    )
    parser.add_argument(
        "--pair-manifest",
        type=Path,
        default=ROOT / "outputs/reports/chipbench_holdout_pair_manifest_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chipbench_holdout_orfs_manifest_v1.json",
    )
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def platform_from_config(config: str) -> str:
    parts = Path(config).parts
    index = parts.index("designs")
    return parts[index + 1]


def main() -> None:
    args = parse_args()
    args.verified_manifest = args.verified_manifest.resolve()
    args.pair_manifest = args.pair_manifest.resolve()
    args.output = args.output.resolve()
    verified = load(args.verified_manifest)
    pairs = load(args.pair_manifest)
    pair_checks = pairs.get("checks", {})
    pair_gate = {
        "pair_status_ready": pairs.get("status") == "frozen_pair_manifest_ready",
        "all_routes_matched": bool(pairs.get("all_routes_matched", False)),
        "all_legalized_feasible": bool(
            pairs.get("all_legalized_feasible", False)
        ),
        "all_zero_overlap": bool(pairs.get("all_zero_overlap", False)),
        "all_tcl_distinct": bool(
            pairs.get("all_candidate_control_tcl_distinct", False)
        ),
        "row_checks_present": bool(pairs.get("rows")) and all(
            bool(row.get("checks", {}).get("distinct_tcl", False))
            for row in pairs.get("rows", [])
        ),
        "embedded_checks_consistent": all(
            bool(pair_checks.get(name, False))
            for name in (
                "all_distinct",
                "all_legalized",
                "all_zero_overlap",
                "all_routes_matched",
            )
            if name in pair_checks
        ),
    }
    if not all(pair_gate.values()):
        report = {
            "version": "chipbench-holdout-orfs-manifest-v1",
            "status": "blocked_invalid_placement_pairs",
            "inventory": [],
            "selection": {
                "selected_cases": [],
                "independent_designs": 0,
            },
            "pair_gate": pair_gate,
            "source": {
                "verified_manifest": display_path(args.verified_manifest),
                "pair_manifest": display_path(args.pair_manifest),
            },
            "claim_boundary": (
                "No ORFS route cohort may be staged from collapsed, illegal, "
                "overlapping or route-mismatched candidate/control pairs."
            ),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, indent=2))
        raise SystemExit(1)
    pair_cases = {row["case"] for row in pairs["rows"]}
    inventory = []
    for row in verified["designs"]:
        if row["id"] not in pair_cases:
            continue
        config = row["flow_config"]
        nickname = row.get("design_nickname", row["id"].split("/")[-1])
        inventory.append(
            {
                "case": row["id"],
                "design_key": row["id"],
                "config": config,
                "platform": (
                    row.get("platform")
                    or platform_from_config(config)
                ),
                "design_name": row.get("design_name", nickname),
                "design_nickname": nickname,
            }
        )
    if len(inventory) != len(pair_cases):
        raise ValueError("verified/pair case coverage mismatch")
    report = {
        "version": "chipbench-holdout-orfs-manifest-v1",
        "status": "staged_before_routed_ppa",
        "inventory": inventory,
        "pair_gate": pair_gate,
        "selection": {
            "selected_cases": sorted(pair_cases),
            "independent_designs": len(pair_cases),
        },
        "source": {
            "verified_manifest": str(args.verified_manifest.relative_to(ROOT)),
            "pair_manifest": str(args.pair_manifest.relative_to(ROOT)),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "designs": len(inventory),
                "platforms": sorted({row["platform"] for row in inventory}),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
