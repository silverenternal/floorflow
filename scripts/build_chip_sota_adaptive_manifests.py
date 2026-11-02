#!/usr/bin/env python3
"""Freeze disjoint calibration and validation manifests for adaptive refinement."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "outputs/reports/chip_dev_screen_v45.json",
    )
    parser.add_argument(
        "--exclude",
        type=Path,
        nargs="+",
        default=[ROOT / "outputs/reports/chip_sota_refiner_budget_screen8_v1.json"],
    )
    parser.add_argument("--salt", default="chip-sota-adaptive-inference-v1")
    parser.add_argument("--strata", type=int, default=8)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/reports/chip_sota_adaptive_manifests_v1",
    )
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def manifest_indices(payload: dict) -> set[int]:
    values = payload.get("indices", payload)
    return {int(value) for value in values}


def stable_rank(salt: str, case_index: int) -> str:
    return hashlib.sha256(f"{salt}:{case_index}".encode()).hexdigest()


def split_strata(rows: list[dict], count: int) -> list[list[dict]]:
    ordered = sorted(rows, key=lambda row: (int(row["block_count"]), int(row["index"])))
    if len(ordered) != count * 4:
        raise ValueError(f"expected {count * 4} source rows, found {len(ordered)}")
    return [ordered[start : start + 4] for start in range(0, len(ordered), 4)]


def main() -> None:
    args = parse_args()
    source = load(args.source)
    excluded = set().union(*(manifest_indices(load(path)) for path in args.exclude))
    rows = [row for row in source["rows"] if int(row["index"]) not in excluded]
    strata = split_strata(rows, args.strata)

    calibration = []
    validation = []
    stratum_rows = []
    for stratum_id, candidates in enumerate(strata):
        ranked = sorted(candidates, key=lambda row: stable_rank(args.salt, int(row["index"])))
        calibration.append(ranked[0])
        validation.append(ranked[1])
        stratum_rows.append(
            {
                "stratum": stratum_id,
                "candidates": [
                    {
                        "index": int(row["index"]),
                        "block_count": int(row["block_count"]),
                        "rank": stable_rank(args.salt, int(row["index"])),
                    }
                    for row in ranked
                ],
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    common = {
        "source": str(args.source),
        "excluded_indices": sorted(excluded),
        "salt": args.salt,
        "selection_rule": (
            "Exclude every previously opened refiner index. Sort remaining source "
            "cases by block count, split into eight contiguous four-case strata, "
            "rank within each stratum by SHA256(salt:index), and assign the two "
            "lowest-ranked cases to calibration and validation respectively. "
            "No model output or target metric is read."
        ),
        "strata": stratum_rows,
    }
    calibration_payload = {
        "version": "chip-sota-adaptive-calibration-v1",
        "role": "calibration",
        **common,
        "cases": len(calibration),
        "indices": [int(row["index"]) for row in calibration],
        "rows": calibration,
    }
    validation_payload = {
        "version": "chip-sota-adaptive-validation-v1",
        "role": "validation",
        **common,
        "cases": len(validation),
        "indices": [int(row["index"]) for row in validation],
        "rows": validation,
    }
    calibration_path = args.output_dir / "calibration8.json"
    validation_path = args.output_dir / "validation8.json"
    calibration_path.write_text(
        json.dumps(calibration_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    validation_path.write_text(
        json.dumps(validation_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "calibration": str(calibration_path),
                "validation": str(validation_path),
                "calibration_indices": calibration_payload["indices"],
                "validation_indices": validation_payload["indices"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
