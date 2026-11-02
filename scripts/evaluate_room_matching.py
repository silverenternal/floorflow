#!/usr/bin/env python3
"""Evaluate expected room matching and an optional read-only ceiling diagnostic."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch

from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.flow.core import decode_floorplans, sample_floorplans
from floorflow.geometry.core import compose_known_exterior
from floorflow.representation.core import hungarian_room_iou, reference_distribution_distance
from floorflow.training.checkpoint import load_checkpoint_model
from floorflow.inference.sampling_policy import SamplingPolicy, expected_sample_statistics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=REPO_ROOT / "outputs/cache/cubicasa5k_full_v1")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / ".data/cubicasa5k")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument(
        "--diagnostic-ceiling",
        action="store_true",
        help="mark best-of-N as a non-primary ceiling diagnostic",
    )
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--semantic-decode-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs/eval/room_matching/report.json")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    purpose = "diagnostic_ceiling" if args.diagnostic_ceiling else ("distribution" if args.samples > 1 else "single_sample")
    policy = SamplingPolicy(purpose=purpose, candidates_per_condition=max(1, args.samples), seed=args.seed)
    model, config, taxonomy, _state = load_checkpoint_model(args.checkpoint, args.device)
    dataset = CubiCasaFloorplanDataset(
        args.data_root, args.split, image_size=int(config.get("image_size", 256)), cache_root=args.cache_root, limit=args.limit
    )
    rows = []
    for index in range(len(dataset)):
        target, footprint, floor_id, name = dataset[index]
        candidates = []
        area_errors = []
        for _ in range(max(1, args.samples)):
            sampled, _ = sample_floorplans(
                model,
                footprint[None].to(args.device),
                steps=args.steps,
                floor_ids=torch.tensor([floor_id], device=args.device),
                objective_mode=config.get("objective_mode", "velocity"),
                semantic_decode_weight=args.semantic_decode_weight,
            )
            labels = compose_known_exterior(
                decode_floorplans(sampled),
                footprint[None].to(args.device),
                opening_ids=taxonomy.opening_ids,
            )[0].cpu().numpy()
            candidates.append(hungarian_room_iou(labels, target.numpy(), taxonomy)["matched_iou"])
            area_errors.append(
                reference_distribution_distance(labels, target.numpy(), taxonomy)
            )
        statistics = expected_sample_statistics(np.asarray(candidates))
        rows.append(
            {
                "case": name,
                "top1": candidates[0],
                "expected": statistics["mean"],
                "median": statistics["median"],
                "standard_error": statistics["standard_error"],
                "best": max(candidates) if args.diagnostic_ceiling else None,
                "candidates": candidates if args.diagnostic_ceiling else None,
                "area_mae": float(np.mean(area_errors)),
            }
        )
    report = {
        "claim_role": "room_matching_distribution" if not args.diagnostic_ceiling else "room_matching_ceiling_diagnostic",
        "samples": args.samples,
        "sampling_policy": policy.manifest(),
        "top1_hungarian_iou": float(np.mean([row["top1"] for row in rows])),
        "expected_hungarian_iou": float(np.mean([row["expected"] for row in rows])),
        "median_hungarian_iou": float(np.mean([row["median"] for row in rows])),
        "best_of_n_hungarian_iou": (
            float(np.mean([row["best"] for row in rows]))
            if args.diagnostic_ceiling
            else None
        ),
        "best_of_n_usage": "diagnostic_only_not_primary_claim" if args.diagnostic_ceiling else "not_used",
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "samples",
                    "top1_hungarian_iou",
                    "expected_hungarian_iou",
                    "best_of_n_hungarian_iou",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
