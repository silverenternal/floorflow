#!/usr/bin/env python3
"""Materialize target-free stage-2 controls from the distribution model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from floorflow.data.control_cache import control_cache_path, write_control_cache
from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.distribution.checkpoint import restore_distribution_model
from floorflow.heads.coarse_furniture_refiner import coarse_furniture_control
from floorflow.heads.coarse_structure_refiner import coarse_structure_control
from floorflow.heads.coarse_support_refiner import coarse_support_control
from floorflow.heads.room_program import room_presence_target
from floorflow.inference.distribution_overlay import DistributionOverlaySettings
from floorflow.inference.overlay_decoder import decode_overlay
from floorflow.representation.contour_memory import ContourPartitionMemory


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path(".data/cubicasa5k"))
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("outputs/cache/cubicasa5k_full_v2"),
    )
    parser.add_argument(
        "--retrieval-weights",
        type=Path,
        default=Path("outputs/reports/a1_retrieval_weights_calib191.json"),
    )
    parser.add_argument("--splits", nargs="+", default=("train", "val"))
    parser.add_argument("--limit-train", type=int, default=1024)
    parser.add_argument("--limit-val", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


@torch.inference_mode()
def main():
    args = parse_args()
    device = torch.device(args.device)
    model, taxonomy, _ = restore_distribution_model(args.checkpoint, device)
    memory = ContourPartitionMemory.from_cache(args.cache_root, top_k=1)
    retrieval_weights = torch.tensor(
        json.loads(args.retrieval_weights.read_text(encoding="utf-8"))["weights"],
        device=device,
    )
    settings = DistributionOverlaySettings(opening_element_weight=0.0)
    totals = {}
    for split in args.splits:
        limit = args.limit_train if split == "train" else args.limit_val
        dataset = CubiCasaFloorplanDataset(
            args.data_root,
            split,
            image_size=256,
            cache_root=args.cache_root,
            limit=limit,
            augment=False,
        )
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
        )
        totals[split] = 0
        for _target, footprint, floor_ids, names in loader:
            footprint = footprint[:, None].to(device)
            floor_ids = floor_ids.to(device)
            style_ids = torch.zeros(1, dtype=torch.long, device=device)
            decoded = decode_overlay(
                model,
                memory,
                retrieval_weights,
                footprint,
                taxonomy,
                floor_ids=floor_ids,
                style_ids=style_ids,
                retrieval_prior_weight=0.0,
                settings=settings,
            )
            rooms = decoded["layers"].room_labels
            room_logits = torch.stack(
                [
                    (rooms == class_id).float() * 4.0 - 2.0
                    for class_id in taxonomy.room_ids
                ],
                dim=1,
            )
            program = room_presence_target(rooms, taxonomy)
            write_control_cache(
                control_cache_path(args.output_root, split, names[0]),
                room_logits=room_logits,
                program=program,
                structure=coarse_structure_control(
                    rooms,
                    footprint,
                    taxonomy,
                    size=128,
                ),
                furniture=coarse_furniture_control(
                    rooms,
                    footprint,
                    taxonomy,
                    size=192,
                ),
                support=coarse_support_control(
                    rooms,
                    footprint,
                    taxonomy,
                    size=192,
                ),
            )
            totals[split] += 1
    print(json.dumps({"split_counts": totals}, indent=2))


if __name__ == "__main__":
    main()
