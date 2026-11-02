#!/usr/bin/env python3
"""Materialize frozen CAD wall-candidate cache shards without scorer training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_raster_predicted_control_bridge import RasterPredictedControlDataset
from train_residential_predicted_refiner import _predicted_controls
from train_vector_topology_scaffold_v1 import GRAPH_STAGES

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.inference.cad_stage_runtime import CadStageRuntime
from floorflow.model.predicted_channel_checkpoint import (
    build_predicted_channel_model,
    load_predicted_channel_state_dict,
)
from floorflow.training.cad_wall_candidate_epoch_cache import (
    load_candidate_batch,
    prepare_candidate_cache,
)

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer-checkpoint", type=Path, required=True)
    parser.add_argument("--predicted-root", type=Path, required=True)
    parser.add_argument("--candidate-cache-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / ".data/cubicasa5k")
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=ROOT / "outputs/cache/cubicasa5k_full_v2_384",
    )
    parser.add_argument(
        "--geometry-root",
        type=Path,
        default=ROOT / "outputs/cache/cubicasa5k_geometry_v1",
    )
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--minimum-room-area", type=int, default=64)
    parser.add_argument("--max-walls", type=int, default=128)
    parser.add_argument(
        "--candidate-mode",
        choices=("boundary", "host", "skeleton_portfolio"),
        default="boundary",
    )
    parser.add_argument("--room-program-bias", type=float, default=0.0)
    parser.add_argument("--use-junction-stage", action="store_true")
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    payload = torch.load(args.layer_checkpoint, map_location="cpu", weights_only=False)
    config = payload["config"]
    taxonomy = FloorplanTaxonomy.from_json(payload["taxonomy"])
    prepare_candidate_cache(
        args.candidate_cache_root,
        args,
        args.layer_checkpoint,
    )
    layer_model = build_predicted_channel_model(taxonomy, config).to(device)
    load_predicted_channel_state_dict(layer_model, payload["model"])
    layer_model.requires_grad_(False).eval()
    stage_runtime = CadStageRuntime(
        taxonomy,
        device,
        room_checkpoint=config["room_checkpoint"],
        structure_checkpoint=config["structure_checkpoint"],
        centerline_checkpoint=config["centerline_checkpoint"],
        opening_checkpoint=config["opening_checkpoint"],
        room_blend=float(config["room_blend"]),
        structure_size=128,
        room_program_bias=args.room_program_bias,
        use_junction_stage=args.use_junction_stage,
    )
    dataset = RasterPredictedControlDataset(
        args.predicted_root,
        args.split,
        args.cache_root,
        args.data_root,
        image_size=int(config["image_size"]),
    )
    start = max(0, int(args.offset))
    stop = len(dataset) if args.limit is None else min(len(dataset), start + args.limit)
    if start >= stop:
        raise ValueError(f"empty shard range [{start}, {stop}) for {len(dataset)} cases")
    shard = Subset(dataset, range(start, stop))
    loader = DataLoader(
        shard,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    processed = 0
    names_seen = []
    for _target, footprint, floor_ids, names, image in loader:
        load_candidate_batch(
            cache={},
            split=args.split,
            names=names,
            root=args.predicted_root,
            args=args,
            device=device,
            footprint=footprint.to(device),
            image=image,
            floor_ids=floor_ids,
            taxonomy=taxonomy,
            stage_runtime=stage_runtime,
            layer_model=layer_model,
            controls_loader=_predicted_controls,
            graph_stages=GRAPH_STAGES,
            minimum_room_area=args.minimum_room_area,
            max_walls=args.max_walls,
            disk_root=args.candidate_cache_root,
        )
        processed += len(names)
        names_seen.extend(str(name) for name in names)
        print(f"materialized {processed}/{stop - start}", flush=True)
    report = {
        "version": "cad-wall-candidate-materialization-v1",
        "status": "complete",
        "split": args.split,
        "dataset_cases": len(dataset),
        "offset": start,
        "stop": stop,
        "processed_cases": processed,
        "first_case": names_seen[0] if names_seen else None,
        "last_case": names_seen[-1] if names_seen else None,
        "candidate_cache_root": str(args.candidate_cache_root),
        "layer_checkpoint": str(args.layer_checkpoint),
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
