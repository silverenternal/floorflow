#!/usr/bin/env python3
"""Measure exact wall-time gains from block-sharded Official-100 evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate-shard",
        type=Path,
        nargs="+",
        default=[
            ROOT
            / "outputs/reports/chip_m27_nfe4_ref3200_official100_seed19_low_v1.json",
            ROOT
            / "outputs/reports/chip_m27_nfe4_ref3200_official100_seed19_high_v1.json",
        ],
    )
    parser.add_argument(
        "--trinity-shard",
        type=Path,
        nargs="+",
        default=[
            ROOT / "outputs/reports/chip_trinity_d192_nfe4_ref3200_official100_low_v1.json",
            ROOT / "outputs/reports/chip_trinity_d192_nfe4_ref3200_official100_high_v1.json",
        ],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_operator_profile_v2.json",
    )
    parser.add_argument("--case-key", default="nfe4_refine3200")
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def candidate_timing(paths: list[Path], case_key: str) -> dict:
    reports = [load(path) for path in paths]
    rows = []
    for path, report in zip(paths, reports, strict=True):
        case_path = path.with_name(f"{path.stem}_cases.json")
        rows.extend(load(case_path)[case_key])
    indices = sorted(int(row["index"]) for row in rows)
    if len(indices) != 100 or len(set(indices)) != 100:
        raise ValueError("candidate shards must cover each Official-100 index once")
    wall = [float(report["results"][case_key]["wall_clock_seconds"]) for report in reports]
    return {
        "cases": len(rows),
        "shard_wall_seconds": wall,
        "serial_wall_seconds": sum(wall),
        "parallel_wall_seconds": max(wall),
        "speedup": sum(wall) / max(wall),
        "stage_seconds": {
            name: sum(
                float(report["results"][case_key]["stage_seconds"][name])
                for report in reports
            )
            for name in reports[0]["results"][case_key]["stage_seconds"]
        },
    }


def trinity_timing(paths: list[Path]) -> dict:
    reports = [load(path) for path in paths]
    indices = sorted(
        int(row["index"]) for report in reports for row in report["rows"]
    )
    if len(indices) != 100 or len(set(indices)) != 100:
        raise ValueError("Trinity shards must cover each Official-100 index once")
    wall = [
        sum(float(row["seconds"]) for row in report["rows"])
        for report in reports
    ]
    return {
        "cases": len(indices),
        "shard_wall_seconds": wall,
        "serial_wall_seconds": sum(wall),
        "parallel_wall_seconds": max(wall),
        "speedup": sum(wall) / max(wall),
    }


def main() -> None:
    args = parse_args()
    candidate = candidate_timing(args.candidate_shard, args.case_key)
    trinity = trinity_timing(args.trinity_shard)
    report = {
        "version": "chip-operator-profile-v2",
        "status": (
            "pass"
            if candidate["speedup"] >= 1.1 and trinity["speedup"] >= 1.1
            else "fail"
        ),
        "scope": (
            "Exact block-range sharding of Official-100. Model, NFE, refiner, "
            "legalizer and per-case RNG are unchanged."
        ),
        "candidate": candidate,
        "trinity": trinity,
        "interpretation": (
            "Wall speedup is the sum of independently measured shard times divided "
            "by the observed concurrent maximum. It measures scheduler utilization, "
            "not a changed neural operator."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
