#!/usr/bin/env python3
"""Audit pixel-distribution room priors on a frozen contour backbone."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.evaluation.room_metric_suite import room_metric_suite
from floorflow.model.distribution_controller import PixelDistributionController
from floorflow.training.checkpoint import load_checkpoint_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--distribution-checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path(".data/cubicasa5k"))
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("outputs/cache/cubicasa5k_full_v2"),
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--scales", nargs="+", type=float, default=[0.0, 0.1, 0.5, 1.0])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    base, config, taxonomy, _ = load_checkpoint_model(
        args.base_checkpoint,
        device,
        taxonomy=None,
    )
    base.eval().requires_grad_(False)
    controller = PixelDistributionController(
        base,
        num_classes=taxonomy.num_classes,
        room_ids=taxonomy.room_ids,
        hidden_channels=96,
    ).to(device)
    head_payload = torch.load(
        args.distribution_checkpoint,
        map_location=device,
        weights_only=False,
    )
    controller.distribution.load_state_dict(head_payload["model"])
    controller.distribution.eval()
    dataset = CubiCasaFloorplanDataset(
        args.data_root,
        args.split,
        image_size=int(config.get("image_size", 256)),
        cache_root=args.cache_root,
        limit=args.limit,
        augment=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    room_ids = torch.tensor(taxonomy.room_ids, device=device)
    report = {}
    for scale in args.scales:
        controller.prior_scale.fill_(float(scale))
        rows = []
        for target, footprint, floor_ids, _names in loader:
            target = target.to(device)
            footprint = footprint.to(device)
            floor_ids = floor_ids.to(device)
            x_t = torch.zeros(
                target.shape[0],
                base.target_channels,
                *target.shape[-2:],
                device=device,
            )
            _velocity, auxiliary = controller.forward_with_aux(
                x_t,
                torch.zeros(target.shape[0], device=device),
                footprint,
                floor_ids=floor_ids,
            )
            labels = room_ids[
                auxiliary["room_partition_logits"].argmax(dim=1)
            ] * footprint.long()
            for index in range(target.shape[0]):
                support = torch.isin(labels[index], room_ids)
                suite = room_metric_suite(
                    labels[index].cpu().numpy(),
                    target[index].cpu().numpy(),
                    footprint[index].cpu().numpy(),
                    taxonomy,
                    predicted_room_support=support.cpu().numpy(),
                    predicted_instance_support=support.cpu().numpy(),
                    target_room_support=torch.isin(
                        target[index],
                        room_ids,
                    ).cpu().numpy(),
                )
                instance = suite["matched_instance"]
                rows.append(
                    {
                        "room_macro_iou": suite["visible_semantic"]["room_macro_iou"],
                        "instance_geometric_iou": instance["matched_geometric_iou"],
                        "instance_room_f1": instance["instance_f1_at_threshold"],
                        "instance_type_accuracy": instance["matched_type_accuracy"],
                    }
                )
        report[str(scale)] = {
            key: float(np.mean([row[key] for row in rows]))
            for key in rows[0]
        }
        print(json.dumps({"scale": scale, **report[str(scale)]}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "cases": len(dataset),
                "condition_mode": "contour_only",
                "results": report,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
