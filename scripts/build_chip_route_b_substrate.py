#!/usr/bin/env python3
"""Audit the existing FloorSet data, constraint, and scorer substrate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from floorflow.chip_flow.cache import PackedCaseStore

ROOT = Path(__file__).resolve().parents[1]


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    packed_root = ROOT / "outputs/cache/chip_floorset_50k_train_12k_dev_v2"
    packed = {
        split: PackedCaseStore(packed_root / split).metadata
        for split in ("train", "dev")
    }
    split = read(ROOT / "outputs/reports/chip_floorset_split_v1.json")
    physics = read(ROOT / "outputs/reports/chip_physics_parity_v1.json")
    official = read(
        ROOT
        / "outputs/chip_large/v2_20k/dense_no_ot/"
        "official_test100_nfe8_legalized_parallel_check_v2.json"
    )
    sources = [
        "src/floorflow/chip_flow/floorset.py",
        "src/floorflow/chip_flow/cache.py",
        "src/floorflow/chip_physics/terms.py",
        "scripts/evaluate_chip_official.py",
    ]
    checks = {
        "official_floor_set_indexed": split.get("layouts_indexed") == 1008000,
        "packed_train_50000": packed["train"]["cases"] == 50000,
        "packed_dev_at_least_12000": packed["dev"]["cases"] >= 12000,
        "feature_dim_10": all(item["feature_dim"] == 10 for item in packed.values()),
        "all_constraint_checks_agree": all(
            value == 1.0 for value in physics["agreement"].values()
        ),
        "official_test100_scored": official.get("cases") == 100,
        "official_legalized_feasible": (
            official.get("legalized_feasible_fraction") == 1.0
        ),
    }
    report = {
        "status": "pass" if all(checks.values()) else "fail",
        "route": "B",
        "substrate": {
            "official_floor_set": {
                "shards": split["shards"],
                "layouts": split["layouts_indexed"],
                "train_cases": split["train_cases"],
                "dev_cases": split["dev_cases"],
                "block_range": [
                    split["train_min_blocks"],
                    split["train_max_blocks"],
                ],
            },
            "packed_splits": {
                split_name: {
                    "cases": metadata["cases"],
                    "block_range": [
                        metadata["min_blocks"],
                        metadata["max_blocks"],
                    ],
                    "feature_dim": metadata["feature_dim"],
                }
                for split_name, metadata in packed.items()
            },
            "constraint_schema": [
                "fixed",
                "preplaced",
                "MIB group",
                "cluster",
                "boundary",
            ],
            "physics_parity": {
                "agreement": physics["agreement"],
                "positive_cases": physics["official_positive_cases"],
            },
            "official_scorer": {
                "cases": official["cases"],
                "legalized_hard_cost": official["legalized_hard_cost"],
                "legalized_v_rel": official["legalized_v_rel"],
                "legalized_feasible_fraction": official[
                    "legalized_feasible_fraction"
                ],
            },
        },
        "checks": checks,
        "sources": {
            source: digest(ROOT / source) for source in sources
        },
        "reuse_note": (
            "The route-B substrate already exists. Extend these adapters and "
            "evaluation scripts rather than creating another FloorSet pipeline."
        ),
    }
    output = ROOT / "outputs/reports/chip_route_b_substrate_v1.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
