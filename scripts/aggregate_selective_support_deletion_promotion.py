#!/usr/bin/env python3
"""Aggregate the two-seed selective support promotion evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEYS = ("room_iou", "instance_f1", "count_mae", "room_support_iou")
LOWER_IS_BETTER = {"count_mae"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[19, 20])
    parser.add_argument(
        "--train-root",
        type=Path,
        default=ROOT / "outputs/train",
    )
    parser.add_argument(
        "--report-root",
        type=Path,
        default=ROOT / "outputs/reports",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/selective_support_deletion_promotion_twoseed_v1.json",
    )
    return parser.parse_args()


def _improved(delta: float, key: str) -> bool:
    return delta < 0.0 if key in LOWER_IS_BETTER else delta > 0.0


def _calibration_seed(seed: int, root: Path) -> dict:
    path = (
        root
        / f"selective_support_deletion_blend50_promotion1024_seed{seed}_v1"
        / "report.json"
    )
    report = json.loads(path.read_text(encoding="utf-8"))
    row = max(report["history"], key=lambda value: value["selection_score"])
    baseline = row["validation"]["blend50"]
    candidate = row["validation"]["selective_deletion"]
    return {
        "path": str(path),
        "epoch": int(row["epoch"]),
        "baseline": baseline,
        "candidate": candidate,
        "delta": {
            key: float(candidate[key] - baseline[key])
            for key in KEYS
        },
    }


def _evaluation_seed(seed: int, split: str, root: Path) -> dict:
    path = (
        root
        / f"selective_support_deletion_seed{seed}_{split}_v1.json"
    )
    report = json.loads(path.read_text(encoding="utf-8"))
    return {
        "path": str(path),
        "cases": int(report["validation_cases"]),
        "baseline": report["summary"]["blend50"],
        "candidate": report["summary"]["selective"],
        "paired": {
            key: report["paired"][key]
            for key in KEYS
        },
        "passed": bool(report["gates"]["passed"]),
    }


def main() -> None:
    args = parse_args()
    splits = {
        "cal256": {
            str(seed): _calibration_seed(seed, args.train_root)
            for seed in args.seeds
        },
        "holdout128_offset256": {
            str(seed): _evaluation_seed(seed, "holdout256", args.report_root)
            for seed in args.seeds
        },
        "val491": {
            str(seed): _evaluation_seed(seed, "val491", args.report_root)
            for seed in args.seeds
        },
    }
    primary_gate = {}
    for split in ("cal256", "holdout128_offset256", "val491"):
        primary_gate[split] = {}
        for key in KEYS:
            improved = 0
            for row in splits[split].values():
                delta = (
                    row["delta"][key]
                    if "delta" in row
                    else row["paired"][key]["mean_delta"]
                )
                improved += int(_improved(float(delta), key))
            primary_gate[split][key] = {
                "improved_seeds": improved,
                "non_regression_seeds": improved,
                "passed": improved == len(args.seeds),
            }
    report = {
        "schema_version": "selective-support-deletion-promotion-v1",
        "protocol": "predicted_control",
        "test_opened": False,
        "target_used_at_inference": False,
        "seeds": args.seeds,
        "splits": splits,
        "gates": primary_gate,
        "promotion_passed": all(
            value["passed"]
            for split in primary_gate.values()
            for value in split.values()
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
