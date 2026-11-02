#!/usr/bin/env python3
"""Build a fail-closed route-transfer provenance and freshness manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--calibration",
        type=Path,
        default=ROOT / "outputs/reports/chip_external_ppa_holdout_v44_all.json",
    )
    parser.add_argument(
        "--observed-v45",
        type=Path,
        default=ROOT / "outputs/reports/chip_v45_repaired_cohort_manifest.json",
    )
    parser.add_argument(
        "--chipbench-pairs",
        type=Path,
        default=ROOT / "outputs/reports/chipbench_holdout_pair_manifest_v1.json",
    )
    parser.add_argument(
        "--chipbench-verified",
        type=Path,
        default=ROOT / "outputs/reports/chip_chipbench_holdout_manifest_v1_verified.json",
    )
    parser.add_argument(
        "--orfs-inventory",
        type=Path,
        default=ROOT / "outputs/reports/chip_external_benchmark_manifest_v1.json",
    )
    parser.add_argument(
        "--orfs-candidates",
        type=Path,
        default=ROOT / "outputs/reports/chip_orfs_macro_candidate_scan_v1.json",
    )
    parser.add_argument(
        "--lock",
        type=Path,
        default=ROOT / "configs/chip_orfs_external_lock_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_route_transfer_provenance_v2.json",
    )
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def canonical_family(case: str, source_name: str | None = None) -> str:
    aliases = {
        "ariane81": "ariane",
        "ariane133": "ariane",
        "ariane136": "ariane",
        "bp68": "black_parrot",
        "bp": "black_parrot",
        "bp_be12": "bp_be_top",
        "bp_be": "bp_be_top",
        "bp_fe38": "bp_fe_top",
        "bp_fe": "bp_fe_top",
        "bp_multi57": "bp_multi_top",
        "bp_multi": "bp_multi_top",
        "swerv_wrapper43": "swerv_wrapper",
    }
    key = (source_name or case.split("/")[-1]).lower()
    return aliases.get(key, key)


def route_recipe(
    flow: dict[str, Any] | None,
    image_digest: str,
) -> dict[str, str] | None:
    if not flow or not flow.get("command"):
        return None
    shell = str(flow["command"][-1])
    tokens = shell.split()
    recipe = {
        "target": tokens[-1] if tokens else "unknown",
        "or_seed": next(
            (token.split("=", 1)[1] for token in tokens if token.startswith("OR_SEED=")),
            "unknown",
        ),
        "grt_seed": next(
            (token.split("=", 1)[1] for token in tokens if token.startswith("GRT_SEED=")),
            "unknown",
        ),
        "detailed_route_args": next(
            (token for token in tokens if token.startswith("DETAILED_ROUTE_ARGS=")),
            "",
        ),
        "image_manifest_digest": image_digest,
    }
    payload = json.dumps(recipe, sort_keys=True).encode()
    recipe["sha256"] = hashlib.sha256(payload).hexdigest()
    return recipe


def calibration_rows(
    report: dict[str, Any],
    inventory: dict[str, Any],
    image_digest: str,
) -> list[dict[str, Any]]:
    inventory_by_case = {
        row["case"]: row for row in inventory.get("inventory", [])
    }
    runs = {
        (row["case"], int(row["seed"]), row["arm"]): row
        for row in report.get("runs", [])
    }
    rows = []
    for comparison in report.get("comparisons", []):
        case = comparison["case"]
        seed = int(comparison["seed"])
        candidate = runs.get((case, seed, "candidate"))
        control = runs.get((case, seed, "control"))
        if candidate is None or control is None:
            raise ValueError(f"missing calibration arm for {case} seed {seed}")
        candidate_recipe = route_recipe(candidate.get("flow"), image_digest)
        control_recipe = route_recipe(control.get("flow"), image_digest)
        config = inventory_by_case.get(case, {}).get("config")
        rows.append(
            {
                "case": case,
                "source_family": canonical_family(case),
                "seed": seed,
                "config": config,
                "config_sha256": digest(ROOT / config) if config else None,
                "candidate": {
                    "tcl": candidate.get("macro_placement_tcl"),
                    "tcl_sha256": candidate.get("macro_placement_tcl_sha256"),
                    "route_recipe": candidate_recipe,
                },
                "control": {
                    "tcl": control.get("macro_placement_tcl"),
                    "tcl_sha256": control.get("macro_placement_tcl_sha256"),
                    "route_recipe": control_recipe,
                },
                "distinct_tcl": (
                    candidate.get("macro_placement_tcl_sha256")
                    != control.get("macro_placement_tcl_sha256")
                ),
                "route_recipe_recorded": (
                    candidate_recipe is not None and control_recipe is not None
                ),
                "route_recipe_matched": (
                    candidate_recipe is not None
                    and control_recipe is not None
                    and candidate_recipe["sha256"] == control_recipe["sha256"]
                ),
            }
        )
    return rows


def chipbench_rows(
    pairs: dict[str, Any],
    verified: dict[str, Any],
) -> list[dict[str, Any]]:
    verified_by_case = {row["id"]: row for row in verified.get("designs", [])}
    rows = []
    for row in pairs.get("rows", []):
        case = row["case"]
        metadata = verified_by_case.get(case, {})
        config = metadata.get("flow_config")
        rows.append(
            {
                "case": case,
                "source_family": canonical_family(case),
                "seed": 19,
                "config": config,
                "config_sha256": digest(ROOT / config) if config else None,
                "legalizer_route": row.get("legalizer_route"),
                "candidate": row.get("candidate"),
                "control": row.get("control"),
                "distinct_tcl": bool(
                    row.get("checks", {}).get("distinct_tcl", False)
                ),
                "route_recipe_recorded": False,
                "route_recipe_matched": False,
                "routed_pair_complete": False,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    calibration = load(args.calibration)
    observed_v45 = load(args.observed_v45)
    chipbench_pairs = load(args.chipbench_pairs)
    verified = load(args.chipbench_verified)
    inventory = load(args.orfs_inventory)
    candidates = load(args.orfs_candidates)
    lock = load(args.lock)
    image = lock["image"]
    image_digest = image["linux_amd64_manifest_digest"]

    calibration_runs = calibration_rows(calibration, inventory, image_digest)
    observed_families = {
        canonical_family(case) for case in calibration.get("cases", [])
    }
    observed_families.update(
        canonical_family(row["case"]) for row in observed_v45.get("pairs", [])
    )
    opened_chipbench = chipbench_rows(chipbench_pairs, verified)
    opened_families = {row["source_family"] for row in opened_chipbench}

    potential = []
    for row in candidates.get("candidates", []):
        family = canonical_family(row["id"], row.get("design"))
        potential.append(
            {
                "id": row["id"],
                "source_family": family,
                "config": row.get("config"),
                "macro_master_count": row.get("macro_master_count"),
                "observed_case": row["id"] in calibration.get("cases", []),
                "observed_family": family in observed_families,
                "opened_chipbench_family": family in opened_families,
            }
        )
    unopened = [
        row
        for row in potential
        if not row["observed_family"] and not row["opened_chipbench_family"]
    ]

    observed_family_index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in calibration_runs:
        observed_family_index[row["source_family"]].append(
            {"cohort": "v44_calibration", "case": row["case"], "seed": row["seed"]}
        )
    for row in observed_v45.get("pairs", []):
        observed_family_index[canonical_family(row["case"])].append(
            {"cohort": "v45_observed", "case": row["case"], "seed": 19}
        )
    for row in opened_chipbench:
        observed_family_index[row["source_family"]].append(
            {"cohort": "chipbench_opened", "case": row["case"], "seed": 19}
        )

    calibration_valid = [
        row for row in calibration_runs if row["distinct_tcl"]
    ]
    calibration_complete = [
        row for row in calibration_valid if row["route_recipe_matched"]
    ]
    fresh_distinct = [row for row in opened_chipbench if row["distinct_tcl"]]
    fresh_routed = [
        row for row in opened_chipbench if row["routed_pair_complete"]
    ]
    checks = {
        "calibration_pairs_are_distinct": len(calibration_valid)
        == len(calibration_runs),
        "calibration_route_recipe_complete": len(calibration_complete)
        == len(calibration_runs),
        "chipbench_pairs_are_distinct": chipbench_pairs.get("status")
        == "frozen_pair_manifest_ready",
        "at_least_15_to_20_unopened_designs": len(unopened) >= 15,
        "at_least_15_fresh_routed_pairs": len(fresh_routed) >= 15,
    }
    status = (
        "fresh_route_confirmation_ready"
        if all(checks.values())
        else "blocked_no_unopened_routed_confirmation"
    )
    report = {
        "version": "chip-route-transfer-provenance-v2",
        "status": status,
        "image": {
            "local_tag": image["local_tag"],
            "manifest_digest": image_digest,
            "config_digest": image["linux_amd64_config_digest"],
            "archive": image.get("docker_archive"),
            "archive_sha256_recorded": image.get("docker_archive_sha256"),
        },
        "calibration": {
            "source": display_path(args.calibration),
            "independent_designs": len(calibration.get("cases", [])),
            "repeated_measurements": len(calibration_runs),
            "distinct_legalized_pairs": len(calibration_valid),
            "route_recipe_complete_pairs": len(calibration_complete),
            "rows": calibration_runs,
        },
        "opened_chipbench": {
            "source": display_path(args.chipbench_pairs),
            "independent_case_ids": len(opened_chipbench),
            "distinct_legalized_pairs": len(fresh_distinct),
            "routed_pairs": len(fresh_routed),
            "collapsed_cases": chipbench_pairs.get("collapsed_cases", []),
            "rows": opened_chipbench,
        },
        "fresh_capacity": {
            "target_designs": [15, 20],
            "unopened_static_designs": unopened,
            "unopened_design_count": len(unopened),
            "design_shortfall_at_15": max(0, 15 - len(unopened)),
            "observed_source_families": {
                name: rows for name, rows in sorted(observed_family_index.items())
            },
        },
        "checks": checks,
        "decision": (
            "Existing routed outcomes are calibration or sensitivity evidence. "
            "No candidate outcome may be relabeled as a fresh holdout."
            if status != "fresh_route_confirmation_ready"
            else "Provenance and freshness gates pass."
        ),
        "required_next_steps": [
            "Repair the 13 collapsed ChiPBench legalized pairs before any route launch.",
            "Reserve at least 15 source-disjoint, outcome-unopened macro designs.",
            "Record candidate and control TCL hashes, ORFS image digest, seed and route recipe before observing PPA.",
            "Run both arms with identical config, route recipe, seed and hold metrics.",
            "Do not pool repeated seeds as independent designs.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": status,
                "calibration_designs": report["calibration"][
                    "independent_designs"
                ],
                "calibration_complete_pairs": len(calibration_complete),
                "opened_chipbench_pairs": len(opened_chipbench),
                "opened_chipbench_distinct": len(fresh_distinct),
                "unopened_designs": len(unopened),
                "checks": checks,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
