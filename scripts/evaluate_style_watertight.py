#!/usr/bin/env python3
"""Evaluate conditional watertight validity for style-conditioned samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.distribution.checkpoint import restore_distribution_model
from floorflow.distribution.sampling import sample_distribution_floorplans
from floorflow.evaluation.fpeval_bridge import fpeval_raster_metrics
from floorflow.geometry.core import compose_known_exterior


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/train/distribution_style_prompt_r94_v2/best.pt"),
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--style-id", type=int, default=0)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--noise", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    model, taxonomy, _ = restore_distribution_model(args.checkpoint, device)
    dataset = CubiCasaFloorplanDataset(
        ".data/cubicasa5k",
        args.split,
        image_size=256,
        cache_root="outputs/cache/cubicasa5k_full_v2",
        augment=False,
        limit=args.limit,
    )
    style_ids = torch.tensor([args.style_id], device=device, dtype=torch.long)
    rows = []
    for index in range(len(dataset)):
        target, footprint, floor_id, name = dataset[index]
        footprint_tensor = footprint[None, None].to(device)
        state, _ = sample_distribution_floorplans(
            model,
            footprint_tensor,
            floor_ids=torch.tensor([floor_id], device=device),
            style_ids=style_ids,
            steps=args.steps,
            noise_scale=args.noise,
        )
        prediction = state.argmax(dim=1) + 1
        prediction = compose_known_exterior(
            prediction,
            footprint_tensor,
            opening_ids=taxonomy.opening_ids,
        )
        result = fpeval_raster_metrics(
            prediction[0].cpu().numpy(),
            target.numpy(),
            taxonomy,
            tolerance_fractions=(0.015,),
        )
        predicted_validity = result["prediction_validity"]
        target_validity = result["target_validity"]
        rows.append(
            {
                "id": str(name),
                "target_watertight": bool(target_validity["watertight"]),
                "prediction_watertight": bool(
                    predicted_validity["watertight"]
                ),
                "prediction_valid": bool(predicted_validity["valid"]),
                "fpeval": result["summary"]["0.015"],
            }
        )
        print(f"evaluated {index + 1}/{len(dataset)}", flush=True)
    conditional = [
        row["prediction_watertight"]
        for row in rows
        if row["target_watertight"]
    ]
    report = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "cases": len(rows),
        "style_id": args.style_id,
        "best_of_n_used": False,
        "target_used_for_generation": False,
        "conditional_watertight": (
            float(np.mean(conditional)) if conditional else None
        ),
        "unconditional_watertight": float(
            np.mean([row["prediction_watertight"] for row in rows])
        ),
        "validity": float(np.mean([row["prediction_valid"] for row in rows])),
        "target_watertight_fraction": float(
            np.mean([row["target_watertight"] for row in rows])
        ),
        "mean_fpeval": {
            key: float(
                np.mean(
                    [
                        row["fpeval"][key]
                        for row in rows
                        if isinstance(row["fpeval"][key], (int, float))
                    ]
                )
            )
            for key in rows[0]["fpeval"]
            if isinstance(rows[0]["fpeval"][key], (int, float))
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "conditional_watertight": report["conditional_watertight"],
                "unconditional_watertight": report["unconditional_watertight"],
                "validity": report["validity"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
