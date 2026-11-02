#!/usr/bin/env python3
"""Build the S3 capacity-screen plan from measured FLOP preflights."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from floorflow.chip_flow.training_runtime import write_json_atomic

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--budget-report",
        type=Path,
        default=ROOT / "outputs/reports/chip_capacity_budget_v1.json",
    )
    parser.add_argument(
        "--extreme-reports",
        type=Path,
        nargs="+",
        default=[
            ROOT / "outputs/reports/chip_capacity_budget_b21_v1.json",
            ROOT / "outputs/reports/chip_capacity_budget_b120_v1.json",
        ],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "configs/chip_sota_capacity_screen_v1.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "outputs/reports/chip_capacity_screen_plan_v1.json",
    )
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rows_by_width(report: dict) -> dict[int, dict]:
    return {int(row["width"]): row for row in report["rows"]}


def equal_flop_steps(base_steps: int, ratio: float) -> int:
    return max(1, round(base_steps / ratio))


def make_arm(
    width: int,
    base_steps: int,
    budget_class: str,
    ratio: float,
) -> dict:
    steps = (
        equal_flop_steps(base_steps, ratio)
        if budget_class in ("equal-flops", "approximately_equal_training_flops")
        else base_steps
    )
    return {
        "id": f"d{width}-{budget_class}",
        "width": width,
        "steps": steps,
        "budget_class": budget_class,
        "measured_block_ratio": ratio,
        "relative_training_flops": ratio * steps / base_steps,
    }


def build_plan(reports: list[dict], base_steps: int = 20_000) -> dict:
    by_width = [rows_by_width(report) for report in reports]
    block_counts = [int(report["block_count"]) for report in reports]
    ratios = {}
    for width in (256, 384):
        per_block = [
            row[width]["training_forward_backward_flops"]
            / row[192]["training_forward_backward_flops"]
            for row in by_width
        ]
        ratios[str(width)] = {
            "per_block": {
                str(blocks): ratio
                for blocks, ratio in zip(block_counts, per_block, strict=True)
            },
            "mean": statistics.fmean(per_block),
            "min": min(per_block),
            "max": max(per_block),
        }
    def arms_for(base: int) -> list[dict]:
        output = [make_arm(192, base, "control", 1.0)]
        for width in (256, 384):
            ratio = ratios[str(width)]["mean"]
            output.extend(
                [
                    make_arm(width, base, "equal-exposure", ratio),
                    make_arm(width, base, "equal-flops", ratio),
                ]
            )
        return output

    arms = arms_for(base_steps)
    return {
        "version": "chip-sota-capacity-screen-v1",
        "status": "prepared_waiting_for_S1",
        "source_reports": [report["version"] for report in reports],
        "block_counts": block_counts,
        "base_steps": base_steps,
        "flop_ratios": ratios,
        "quick_screen": {
            "train_indices": "configs/chip_train_screen_5k_v1.json",
            "cache": "outputs/cache/chip_floorset_50k_train_12k_dev_v2",
            "base_steps": 5_000,
            "arms": arms_for(5_000),
            "seed": 19,
            "device": "cuda:0",
        },
        "confirmation": {
            "train_indices": "configs/chip_train_scaling_20k_v1.json",
            "cache": "outputs/cache/chip_floorset_50k_train_12k_dev_v2",
            "arms": arms,
            "seed": 19,
            "device": "cuda:0",
        },
        "inference": {
            "primary": "NFE4 + refiner400 + one sample",
            "diagnostic": "NFE4 + refiner1600 + one sample",
        },
        "promotion_rule": {
            "quick": "Run d256 equal-exposure and equal-FLOP arms before d384.",
            "d384": "Open only if d256 improves both cost and V_rel in at least one budget class without worsening the high-block tail.",
            "confirmation": "Promote only a setting that improves both primary metrics under a named equal-exposure or equal-FLOP class.",
        },
        "boundary": "Capacity screen budget only; no full-data or external claim.",
    }


def main() -> None:
    args = parse_args()
    reports = [load(args.budget_report)] + [
        load(path) for path in args.extreme_reports
    ]
    plan = build_plan(reports)
    write_json_atomic(args.output, plan)
    write_json_atomic(args.report, plan)
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
