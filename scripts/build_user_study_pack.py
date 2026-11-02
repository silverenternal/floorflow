#!/usr/bin/env python3
"""Freeze a blinded case and condition-order pack for the user study."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from floorflow.data.dataset import CubiCasaFloorplanDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--num-styles", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reports/user_study_case_pack_v1.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = CubiCasaFloorplanDataset(
        ".data/cubicasa5k",
        args.split,
        image_size=256,
        cache_root="outputs/cache/cubicasa5k_full_v2",
        augment=False,
        limit=args.limit,
    )
    names = [str(dataset[index][3]) for index in range(len(dataset))]
    rng = random.Random(args.seed)
    rng.shuffle(names)
    style_sequence = [
        style
        for style in range(args.num_styles)
        for _ in range((len(names) + args.num_styles - 1) // args.num_styles)
    ][: len(names)]
    rng.shuffle(style_sequence)
    conditions = ["style_conditioned", "no_style", "retrieval"]
    cases = []
    for name, style_id in zip(names, style_sequence):
        order = conditions.copy()
        rng.shuffle(order)
        cases.append(
            {
                "case_id": name,
                "requested_style_id": int(style_id),
                "requested_style_text": f"Style {style_id}",
                "condition_order": order,
            }
        )
    report = {
        "status": "pack_frozen",
        "seed": args.seed,
        "split": args.split,
        "cases": len(cases),
        "conditions": conditions,
        "survey_fields": [
            "pairwise_preference",
            "style_fit_1_to_7",
            "perceived_legality_pass_fail",
            "legality_failure_reason",
            "confidence_1_to_7",
            "preferred_style_condition",
        ],
        "human_data_collected": False,
        "case_order": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "cases", "seed")}, indent=2))


if __name__ == "__main__":
    main()
