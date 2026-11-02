#!/usr/bin/env python3
"""Evaluate room-program set decoders on a fixed validation split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from floorflow.data.class_balance import load_positive_weights
from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.heads.room_program import RoomProgramHead, room_presence_target
from floorflow.heads.room_program_set import (
    RoomProgramSetHead,
    load_room_program_set_state,
    program_set_metrics,
)
from floorflow.training.checkpoint import load_checkpoint_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--program-checkpoint", type=Path, required=True)
    parser.add_argument("--bce-checkpoint", type=Path, default=None)
    parser.add_argument("--blends", nargs="+", type=float, default=(1.0,))
    parser.add_argument("--split", default="val")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=191)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    payload = torch.load(args.program_checkpoint, map_location="cpu", weights_only=False)
    config = payload["config"]
    backbone, data_config, taxonomy, _ = load_checkpoint_model(
        Path(payload["backbone_checkpoint"]),
        device,
        taxonomy=None,
    )
    backbone.eval().requires_grad_(False)
    head = RoomProgramSetHead(
        backbone.output.in_channels,
        len(taxonomy.room_ids),
        hidden_dim=int(config.get("hidden_dim", 96)),
        tokens=int(config.get("tokens", 8)),
        layers=int(config.get("layers", 1)),
        heads=int(config.get("heads", 4)),
        num_queries=int(config.get("queries", 32)),
    ).to(device)
    load_room_program_set_state(head, payload["head"])
    positive = torch.from_numpy(
        load_positive_weights(
            config["cache_root"],
            taxonomy.num_target_channels,
            20.0,
        )
    ).to(device)
    room_weights = positive[
        torch.tensor(
            [class_id - 1 for class_id in taxonomy.room_ids],
            device=device,
        )
    ]
    head.program_prior.copy_(room_weights / room_weights.mean())
    head.eval()
    bce_head = None
    if args.bce_checkpoint is not None:
        bce_payload = torch.load(
            args.bce_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        bce_head = RoomProgramHead(
            backbone.output.in_channels,
            len(taxonomy.room_ids),
        ).to(device)
        bce_head.load_state_dict(bce_payload["head"])
        bce_head.eval()
    dataset = CubiCasaFloorplanDataset(
        config["data_root"],
        args.split,
        image_size=int(config.get("image_size", data_config.get("image_size", 256))),
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
    modes = ("head", "expected", "query_count", "query_mass", "oracle")
    prediction_batches = []
    expected_batches = []
    bce_logits_batches = []
    for target, footprint, floor_ids, _names in loader:
        target = target.to(device)
        zero = torch.zeros(
            target.shape[0],
            backbone.target_channels,
            *target.shape[-2:],
            device=device,
        )
        features = backbone._encode_features(
            zero,
            torch.zeros(target.shape[0], device=device),
            footprint.to(device),
            floor_ids=floor_ids.to(device),
        )
        predictions = head(features)
        expected = room_presence_target(target, taxonomy).bool()
        prediction_batches.append(predictions)
        expected_batches.append(expected)
        if bce_head is not None:
            bce_logits_batches.append(
                bce_head(features)["presence_logits"]
            )
    set_outputs = {
        key: torch.cat([batch[key] for batch in prediction_batches])
        for key in (
            "class_logits",
            "cardinality_logits",
            "presence_logits",
            "presence_rank_logits",
        )
    }
    expected = torch.cat(expected_batches)
    rows = {}
    for blend in args.blends:
        outputs = dict(set_outputs)
        if bce_head is not None and blend < 1.0:
            bce_logits = torch.cat(bce_logits_batches)
            outputs["presence_logits"] = (
                blend * set_outputs["presence_logits"]
                + (1.0 - blend) * bce_logits
            )
            outputs["presence_rank_logits"] = outputs["presence_logits"]
        for mode in modes:
            key = f"{blend:.2f}_{mode}"
            rows[key] = program_set_metrics(
                outputs,
                expected,
                cardinality_method=mode,
            )
    report = {
        "checkpoint": str(args.program_checkpoint),
        "split": args.split,
        "offset": args.offset,
        "cases": stop - args.offset,
        "best_of_n_used": False,
        "oracle_mode": "oracle uses target cardinality for diagnostic ceiling only",
        "metrics": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
