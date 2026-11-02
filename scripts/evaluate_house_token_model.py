#!/usr/bin/env python3
"""Evaluate a room-box flow checkpoint with permutation-invariant bbox metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from floorflow.chip_flow.cache import PackedCaseStore
from floorflow.chip_flow.model import ChipBlockFlowModel, decode_boxes_batch
from floorflow.chip_flow.projections import project_conditioned_latent
from floorflow.chip_flow.slot_model import ChipSlotFlowModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache",
        type=Path,
        default=REPO_ROOT / "outputs/cache/house_room_tokens_v1/val",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sampling-steps", type=int, default=8)
    parser.add_argument("--save-cases", action="store_true")
    parser.add_argument("--use-ema", action="store_true")
    return parser.parse_args()


def load_model(
    checkpoint: dict,
    device: str,
    *,
    use_ema: bool,
) -> ChipBlockFlowModel | ChipSlotFlowModel:
    config = checkpoint["config"]
    if config.get("architecture") == "global_slot":
        model = ChipSlotFlowModel(
            feature_dim=int(config.get("feature_dim", 10)),
            hidden=int(config["hidden"]),
            global_dim=int(config.get("global_dim", config["hidden"])),
            target_layers=int(config.get("target_layers", 4)),
            condition_layers=int(config.get("condition_layers", 2)),
            decoder_layers=int(config.get("decoder_layers", 4)),
            heads=int(config["heads"]),
        )
    else:
        model = ChipBlockFlowModel(
            feature_dim=10,
            hidden=int(config["hidden"]),
            layers=int(config["layers"]),
            heads=int(config["heads"]),
            top_k=int(config["sparse_k"]) or None,
        )
    model.load_state_dict(checkpoint["ema"] if use_ema else checkpoint["model"])
    return model.to(device).eval()


def box_iou(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    intersection_x = torch.minimum(
        left[:, None, 0] + left[:, None, 2],
        right[None, :, 0] + right[None, :, 2],
    ) - torch.maximum(left[:, None, 0], right[None, :, 0])
    intersection_y = torch.minimum(
        left[:, None, 1] + left[:, None, 3],
        right[None, :, 1] + right[None, :, 3],
    ) - torch.maximum(left[:, None, 1], right[None, :, 1])
    intersection = intersection_x.clamp_min(0) * intersection_y.clamp_min(0)
    left_area = left[:, 2] * left[:, 3]
    right_area = right[:, 2] * right[:, 3]
    return intersection / (
        left_area[:, None] + right_area[None, :] - intersection
    ).clamp_min(1.0e-8)


def matched_metrics(predicted: torch.Tensor, target: torch.Tensor) -> dict:
    cost = 1.0 - box_iou(predicted, target)
    rows, columns = linear_sum_assignment(cost.cpu().numpy())
    iou = box_iou(predicted, target)[rows, columns]
    predicted_centers = predicted[:, :2] + predicted[:, 2:] * 0.5
    target_centers = target[:, :2] + target[:, 2:] * 0.5
    center = (predicted_centers[rows] - target_centers[columns]).abs().sum(1)
    size = (predicted[rows, 2:] - target[columns, 2:]).abs().sum(1)
    return {
        "bbox_iou": float(iou.mean()),
        "center_l1": float(center.mean()),
        "size_l1": float(size.mean()),
    }


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> tuple[dict, list[dict]]:
    store = PackedCaseStore(args.cache)
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model = load_model(checkpoint, args.device, use_ema=args.use_ema)
    totals = {"bbox_iou": 0.0, "center_l1": 0.0, "size_l1": 0.0}
    outside = 0
    rows = []
    generator = torch.Generator(device=args.device).manual_seed(19)
    for count in sorted(store.by_block_count):
        indices = store.indices(count)
        for start in range(0, len(indices), args.batch_size):
            batch = store.batch(
                indices[start : start + args.batch_size],
                device=args.device,
            )
            if isinstance(model, ChipSlotFlowModel):
                latent = model.sample(
                    batch["features"],
                    args.sampling_steps,
                    generator=generator,
                )
            else:
                latent = project_conditioned_latent(
                    batch["source_latent"],
                    batch["target_latent"],
                    batch["constraints"],
                )
                for step in range(args.sampling_steps):
                    time = torch.full(
                        (latent.shape[0],),
                        step / args.sampling_steps,
                        device=args.device,
                    )
                    latent = latent + model(
                        latent,
                        time,
                        batch["features"],
                    ) / args.sampling_steps
                    latent = project_conditioned_latent(
                        latent,
                        batch["target_latent"],
                        batch["constraints"],
                    )
            predicted = decode_boxes_batch(latent, batch["areas"])
            for index in range(latent.shape[0]):
                metrics = matched_metrics(
                    predicted[index].cpu(),
                    batch["boxes"][index].cpu(),
                )
                invalid = (
                    (predicted[index, :, :2] < -0.01)
                    | (predicted[index, :, :2] + predicted[index, :, 2:] > 1.01)
                ).any()
                outside += int(invalid)
                row = {
                    "index": int(indices[start + index]),
                    "blocks": count,
                    "outside": bool(invalid),
                    **metrics,
                }
                rows.append(row)
                for name, value in metrics.items():
                    totals[name] += value
    summary = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "cache": str(args.cache),
        "cases": len(rows),
        "sampling_steps": args.sampling_steps,
        "use_ema": args.use_ema,
        **{name: value / len(rows) for name, value in totals.items()},
        "outside_fraction": outside / len(rows),
    }
    return summary, rows


def main() -> None:
    args = parse_args()
    summary, rows = evaluate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.save_cases:
        cases_path = args.output.with_name(f"{args.output.stem}_cases.json")
        cases_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
