#!/usr/bin/env python3
"""Fuse contour-only room-program priors with query-mask class logits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.evaluation.room_metric_suite import room_metric_suite
from floorflow.heads.room_mask_decoder import RoomMaskDecoder
from floorflow.heads.room_mask_fusion import (
    apply_program_class_prior,
    fused_room_mask_logits,
)
from floorflow.heads.room_program_set import RoomProgramSetHead
from floorflow.training.checkpoint import load_checkpoint_model
from floorflow.training.room_mask_finetune import load_room_mask_decoder_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--room-checkpoint", type=Path, required=True)
    parser.add_argument("--program-checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--offset", type=int, default=191)
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--scales", nargs="+", type=float, default=(0.0, 0.25, 0.5, 1.0))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    room_payload = torch.load(args.room_checkpoint, map_location="cpu", weights_only=False)
    program_payload = torch.load(
        args.program_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    room_config = room_payload["config"]
    backbone, data_config, taxonomy, _ = load_checkpoint_model(
        Path(room_payload["backbone_checkpoint"]),
        device,
        taxonomy=None,
    )
    backbone.eval().requires_grad_(False)
    room_model = RoomMaskDecoder(
        backbone.output.in_channels,
        len(taxonomy.room_ids),
        hidden_dim=int(room_config["hidden_dim"]),
        num_queries=int(room_config["queries"]),
        layers=int(room_config["layers"]),
        heads=int(room_config["heads"]),
    ).to(device)
    load_room_mask_decoder_state(room_model, room_payload["head"])
    room_model.eval()
    program_config = program_payload["config"]
    program_model = RoomProgramSetHead(
        backbone.output.in_channels,
        len(taxonomy.room_ids),
        hidden_dim=int(program_config.get("hidden_dim", 96)),
        tokens=int(program_config.get("tokens", 8)),
        layers=int(program_config.get("layers", 1)),
        heads=int(program_config.get("heads", 4)),
        num_queries=int(program_config.get("queries", 32)),
    ).to(device)
    program_model.load_state_dict(program_payload["head"])
    program_model.eval()
    dataset = CubiCasaFloorplanDataset(
        room_config["data_root"],
        args.split,
        image_size=int(room_config.get("image_size", data_config.get("image_size", 256))),
        cache_root=room_config["cache_root"],
        augment=False,
    )
    stop = min(len(dataset), args.offset + args.limit)
    loader = DataLoader(
        Subset(dataset, range(args.offset, stop)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    room_ids = torch.tensor(taxonomy.room_ids, device=device)
    rows = {str(scale): [] for scale in args.scales}
    for target, footprint, floor_ids, _names in loader:
        target = target.to(device)
        footprint = footprint.to(device)
        zero = torch.zeros(
            target.shape[0],
            backbone.target_channels,
            *target.shape[-2:],
            device=device,
        )
        features = backbone._encode_features(
            zero,
            torch.zeros(target.shape[0], device=device),
            footprint,
            floor_ids=floor_ids.to(device),
        )
        predictions = room_model(features)
        program_logits = program_model(features)["presence_logits"]
        for scale in args.scales:
            fused_predictions = dict(predictions)
            fused_predictions["class_logits"] = apply_program_class_prior(
                predictions["class_logits"],
                program_logits,
                scale=scale,
            )
            logits = fused_room_mask_logits(
                fused_predictions,
                len(taxonomy.room_ids),
                fusion_mode="sigmoid",
            )
            prediction = room_ids[logits.argmax(dim=1)] * footprint.long()
            for index in range(target.shape[0]):
                support = torch.isin(prediction[index], room_ids)
                suite = room_metric_suite(
                    prediction[index].cpu().numpy(),
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
                rows[str(scale)].append(
                    {
                        "room_macro_iou": suite["visible_semantic"]["room_macro_iou"],
                        "instance_geometric_iou": instance["matched_geometric_iou"],
                        "instance_room_f1": instance["instance_f1_at_threshold"],
                        "instance_type_accuracy": instance["matched_type_accuracy"],
                    }
                )
    report = {
        "room_checkpoint": str(args.room_checkpoint),
        "program_checkpoint": str(args.program_checkpoint),
        "split": args.split,
        "cases": len(rows[str(args.scales[0])]),
        "best_of_n_used": False,
        "metrics": {
            scale: {
                key: float(np.mean([row[key] for row in values]))
                for key in values[0]
            }
            for scale, values in rows.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
