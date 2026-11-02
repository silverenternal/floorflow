#!/usr/bin/env python3
"""Evaluate contour-derived sparse prior refinement without target controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.distribution.checkpoint import restore_distribution_model
from floorflow.evaluation.room_metric_suite import room_metric_suite
from floorflow.heads.room_mask_decoder import RoomMaskDecoder
from floorflow.heads.room_mask_fusion import fused_room_mask_logits
from floorflow.heads.room_program import gate_room_labels_to_program
from floorflow.inference.predicted_prior import low_confidence_prior_mask
from floorflow.model.sparse_control_attention import (
    MaskedControlFlow,
    sample_sparse_guided_floorplans,
)
from floorflow.training.checkpoint import load_checkpoint_model
from floorflow.training.room_mask_finetune import load_room_mask_decoder_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--distribution-checkpoint", type=Path, required=True)
    parser.add_argument("--sparse-checkpoint", type=Path, required=True)
    parser.add_argument("--room-checkpoint", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=Path(".data/cubicasa5k"))
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("outputs/cache/cubicasa5k_full_v2"),
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--offset", type=int, default=191)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--flow-steps", type=int, default=4)
    parser.add_argument("--hidden-fraction", type=float, default=0.25)
    parser.add_argument("--room-prior-blend", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def room_labels(state, footprint, room_ids):
    labels = room_ids[state[:, room_ids - 1].argmax(dim=1)]
    scores = state[:, room_ids - 1]
    program = scores.amax(dim=(2, 3)) > 0.1
    missing = ~program.any(dim=1)
    if missing.any():
        fallback = scores.amax(dim=(2, 3)).argmax(dim=1)
        program = program.clone()
        program[missing, fallback[missing]] = True
    labels = gate_room_labels_to_program(labels, program, tuple(room_ids.tolist()))
    return labels * footprint.long()


def metric_row(
    prediction,
    target,
    predicted_support,
    target_support,
    footprint,
    taxonomy,
):
    suite = room_metric_suite(
        prediction.cpu().numpy(),
        target.cpu().numpy(),
        footprint.cpu().numpy(),
        taxonomy,
        predicted_room_support=predicted_support.cpu().numpy(),
        predicted_instance_support=predicted_support.cpu().numpy(),
        target_room_support=target_support.cpu().numpy(),
    )
    instance = suite["matched_instance"]
    return {
        "room_macro_iou": suite["visible_semantic"]["room_macro_iou"],
        "instance_geometric_iou": instance["matched_geometric_iou"],
        "instance_room_f1": instance["instance_f1_at_threshold"],
        "instance_type_accuracy": instance["matched_type_accuracy"],
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    distribution, taxonomy, _ = restore_distribution_model(
        args.distribution_checkpoint,
        device,
    )
    distribution.eval()
    sparse_payload = torch.load(
        args.sparse_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    base, config, sparse_taxonomy, _ = load_checkpoint_model(
        args.sparse_checkpoint,
        device,
        taxonomy=taxonomy,
    )
    if sparse_taxonomy.class_names != taxonomy.class_names:
        raise ValueError("distribution and sparse-refiner taxonomies differ")
    attention_type = str(
        sparse_payload.get("config", {}).get("sparse_attention_type", "mita")
    )
    refiner = MaskedControlFlow(base, attention_type=attention_type).to(device)
    refiner.sparse_attention.load_state_dict(
        sparse_payload["sparse_attention"],
        strict=False,
    )
    refiner.eval()
    room_model = None
    if args.room_checkpoint is not None:
        room_payload = torch.load(
            args.room_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        room_config = room_payload["config"]
        room_model = RoomMaskDecoder(
            distribution.base.output.in_channels,
            len(taxonomy.room_ids),
            hidden_dim=int(room_config["hidden_dim"]),
            num_queries=int(room_config["queries"]),
            layers=int(room_config["layers"]),
            heads=int(room_config["heads"]),
        ).to(device)
        load_room_mask_decoder_state(room_model, room_payload["head"])
        room_model.eval()
    dataset = CubiCasaFloorplanDataset(
        args.data_root,
        args.split,
        image_size=int(config.get("image_size", 256)),
        cache_root=args.cache_root,
        limit=args.offset + args.limit,
        augment=False,
    )
    dataset = torch.utils.data.Subset(
        dataset,
        range(args.offset, min(len(dataset), args.offset + args.limit)),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    room_ids = torch.tensor(taxonomy.room_ids, device=device)
    rows = {"raw": [], "refined": []}
    for target, footprint, floor_ids, _names in loader:
        target = target.to(device)
        footprint = footprint[:, None].to(device)
        floor_ids = floor_ids.to(device)
        zero = torch.zeros(
            footprint.shape[0],
            distribution.target_channels,
            *footprint.shape[-2:],
            device=device,
        )
        features = distribution._encode_features(
            zero,
            torch.zeros(footprint.shape[0], device=device),
            footprint,
            floor_ids=floor_ids,
        )
        initial = distribution.distribution_forward(features, footprint)
        prior = initial["cfm"]["state"]
        if room_model is not None and args.room_prior_blend > 0:
            room_probability = fused_room_mask_logits(
                room_model(features),
                len(taxonomy.room_ids),
            ).softmax(dim=1)
            prior = prior.clone()
            room_channels = room_ids - 1
            prior[:, room_channels] = (
                (1.0 - args.room_prior_blend)
                * prior[:, room_channels]
                + args.room_prior_blend * room_probability
            ) * footprint.float()
        control_mask = low_confidence_prior_mask(
            prior,
            footprint,
            hidden_fraction=args.hidden_fraction,
        )
        refined = sample_sparse_guided_floorplans(
            refiner,
            footprint,
            prior,
            control_mask,
            steps=args.flow_steps,
            floor_ids=floor_ids,
            initial_state=prior,
        )
        for name, state in (("raw", prior), ("refined", refined)):
            labels = room_labels(state, footprint[:, 0], room_ids)
            for index in range(labels.shape[0]):
                support = torch.isin(labels[index], room_ids)
                target_support = torch.isin(target[index], room_ids)
                rows[name].append(
                    metric_row(
                        labels[index],
                        target[index],
                        support,
                        target_support,
                        footprint[index, 0],
                        taxonomy,
                    )
                )
    report = {
        "condition_mode": "contour_only",
        "target_control_free": True,
        "best_of_n_used": False,
        "cases": len(rows["raw"]),
        "hidden_fraction": args.hidden_fraction,
        "room_checkpoint": str(args.room_checkpoint) if room_model else None,
        "room_prior_blend": args.room_prior_blend if room_model else 0.0,
        "attention_type": attention_type,
        "metrics": {
            name: {
                key: float(np.mean([row[key] for row in values]))
                for key in values[0]
            }
            for name, values in rows.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
