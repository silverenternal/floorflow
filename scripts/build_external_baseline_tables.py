#!/usr/bin/env python3
"""Build auditable external-baseline result tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.evaluation.external_baseline_metrics import (
    evaluate_housegan_cases,
    evaluate_maskplan_cases,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--data-root", type=Path, default=Path(".data/cubicasa5k"))
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("outputs/cache/cubicasa5k_full_v2"),
    )
    parser.add_argument(
        "--housegan-output",
        type=Path,
        default=Path(
            "outputs/external_baselines/cubicasa_oracle/housegan/output_v2"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reports/external_baseline_comparison_v1.json"),
    )
    parser.add_argument(
        "--maskplan-output",
        type=Path,
        default=Path(
            "third_party/MaskPLAN/Inference/Large_Single_cross/iteration/post"
        ),
    )
    parser.add_argument(
        "--maskplan-boundary-dir",
        type=Path,
        default=Path("third_party/MaskPLAN/parsed_img/img_room_sqe/0"),
    )
    parser.add_argument(
        "--maskplan-test-set",
        type=Path,
        default=Path("third_party/MaskPLAN/Processed_data/Test_set.npy"),
    )
    args = parser.parse_args()
    dataset = CubiCasaFloorplanDataset(
        args.data_root,
        args.split,
        image_size=256,
        cache_root=args.cache_root,
        limit=args.limit,
        augment=False,
    )
    maskplan = evaluate_maskplan_cases(
        args.maskplan_output,
        args.maskplan_boundary_dir,
        np.load(args.maskplan_test_set)[: args.limit],
    )
    report = {
        "status": "complete" if maskplan["cases"] == args.limit else "partial",
        "oracle_program_table": {
            "housediffusion": {
                "status": "adapter_and_forward_validated",
                "note": "58/64 CubiCasa val cases fit the 100-point HouseDiffusion cap",
            },
            "houseganpp": evaluate_housegan_cases(
                dataset,
                args.housegan_output,
                cases=args.limit,
            ),
        },
        "native_table": {
            "maskplan": maskplan
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["oracle_program_table"]["houseganpp"]["summary"], indent=2))


if __name__ == "__main__":
    main()
