#!/usr/bin/env python3
"""Compare room-query fusion modes with one frozen decoder checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.evaluation.room_metric_suite import room_metric_suite
from floorflow.heads.dual_room_mask_decoder import DualPathRoomMaskDecoder
from floorflow.heads.room_mask_decoder import RoomMaskDecoder, fused_room_mask_logits
from floorflow.training.checkpoint import load_checkpoint_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decoder-checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--offset", type=int, default=191)
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=(
            "legacy",
            "sigmoid",
            "query_mixture",
            "hard_query",
            "winner_query",
        ),
        default=(
            "legacy",
            "sigmoid",
            "query_mixture",
            "hard_query",
            "winner_query",
        ),
    )
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    payload = torch.load(
        args.decoder_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    config = payload["config"]
    backbone, _backbone_config, taxonomy, _ = load_checkpoint_model(
        Path(payload["backbone_checkpoint"]),
        device,
        taxonomy=None,
    )
    backbone.eval().requires_grad_(False)
    decoder_type = (
        DualPathRoomMaskDecoder
        if config.get("decoder_mode") == "dual"
        else RoomMaskDecoder
    )
    decoder = decoder_type(
        backbone.output.in_channels,
        len(taxonomy.room_ids),
        hidden_dim=int(config["hidden_dim"]),
        num_queries=int(config["queries"]),
        layers=int(config["layers"]),
        heads=int(config["heads"]),
    ).to(device)
    decoder.load_state_dict(payload["head"], strict=False)
    decoder.eval()
    dataset = CubiCasaFloorplanDataset(
        config["data_root"],
        args.split,
        image_size=int(config.get("image_size", 256)),
        cache_root=config["cache_root"],
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
    rows = {mode: [] for mode in args.modes}
    for target, footprint, floor_ids, _names in loader:
        target = target.to(device)
        footprint = footprint.to(device)
        zeros = torch.zeros(
            target.shape[0],
            backbone.target_channels,
            *target.shape[-2:],
            device=device,
        )
        features = backbone._encode_features(
            zeros,
            torch.zeros(target.shape[0], device=device),
            footprint,
            floor_ids=floor_ids.to(device),
        )
        predictions = decoder(features)
        for mode in args.modes:
            logits = fused_room_mask_logits(
                predictions,
                len(taxonomy.room_ids),
                fusion_mode=mode,
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
                rows[mode].append(
                    {
                        "room_macro_iou": suite["visible_semantic"][
                            "room_macro_iou"
                        ],
                        "instance_geometric_iou": instance[
                            "matched_geometric_iou"
                        ],
                        "instance_room_f1": instance[
                            "instance_f1_at_threshold"
                        ],
                        "instance_type_accuracy": instance[
                            "matched_type_accuracy"
                        ],
                    }
                )
    report = {
        "decoder_checkpoint": str(args.decoder_checkpoint),
        "split": args.split,
        "cases": len(next(iter(rows.values()))),
        "best_of_n_used": False,
        "metrics": {
            mode: {
                key: float(np.mean([row[key] for row in values]))
                for key in values[0]
            }
            for mode, values in rows.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
