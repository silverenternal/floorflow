#!/usr/bin/env python3
"""Audit pin/chart coordinate conventions before enabling route-aware fields."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from floorflow.chip_flow.cache import PackedCaseStore
from floorflow.chip_flow.training_runtime import write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache",
        type=Path,
        default=ROOT / "outputs/cache/chip_official_test_100_v1",
    )
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_route_coordinate_audit_v1.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store = PackedCaseStore(args.cache)
    indices = store.indices()[: args.limit]
    inside = 0
    valid_cases = 0
    single_pin = 0
    span_ratios = []
    min_offsets = []
    for index in indices.tolist():
        batch = store.batch([index])
        count = int(batch["n"])
        pin_count = int(batch["n_pins"][0])
        if pin_count == 0:
            continue
        valid_cases += 1
        pins = batch["pins"][0, :pin_count]
        boxes = batch["boxes"][0, :count]
        lower = boxes[:, :2].amin(dim=0)
        upper = (boxes[:, :2] + boxes[:, 2:]).amax(dim=0)
        inside_mask = (
            pins[:, 0].ge(lower[0])
            & pins[:, 1].ge(lower[1])
            & pins[:, 0].le(upper[0])
            & pins[:, 1].le(upper[1])
        )
        inside += int(inside_mask.all())
        if pin_count == 1:
            single_pin += 1
            continue
        span = float((pins.amax(dim=0) - pins.amin(dim=0)).amax())
        scale = float(batch["areas"][0, :count].sum().sqrt())
        span_ratios.append(span / max(scale, 1.0e-6))
        min_offsets.append(
            float((pins.amin(dim=0) - lower).abs().mean() / max(scale, 1.0e-6))
        )
    report = {
        "version": "chip-route-coordinate-audit-v1",
        "status": "pass" if valid_cases and inside == valid_cases else "review",
        "cache": str(args.cache),
        "cases": valid_cases,
        "all_pins_inside_target_bbox_fraction": inside / max(valid_cases, 1),
        "single_pin_cases": single_pin,
        "pin_span_over_scale": {
            "min": float(np.min(span_ratios)) if span_ratios else None,
            "median": float(np.median(span_ratios)) if span_ratios else None,
            "max": float(np.max(span_ratios)) if span_ratios else None,
        },
        "pin_min_to_bbox_offset_over_scale": {
            "min": float(np.min(min_offsets)) if min_offsets else None,
            "median": float(np.median(min_offsets)) if min_offsets else None,
            "max": float(np.max(min_offsets)) if min_offsets else None,
        },
        "decision": (
            "Pin min/max normalization is required; scale-only normalization "
            "is unsafe when pin origin or span differs from the target chart."
        ),
    }
    write_json_atomic(args.output, report)
    print(args.output)


if __name__ == "__main__":
    main()
