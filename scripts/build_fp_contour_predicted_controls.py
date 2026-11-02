#!/usr/bin/env python3
"""Materialize stage-1 controls from an existing contour-conditional model."""

from __future__ import annotations

import argparse
import json
import random
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from floorflow.data.control_cache import control_cache_path, write_control_cache
from floorflow.data.control_contract import CONTROL_SCHEMA_VERSION, sha256_file
from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.inference.contour_controls import predict_contour_controls
from floorflow.training.checkpoint import load_checkpoint_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path(".data/cubicasa5k"))
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("outputs/cache/cubicasa5k_full_v2"),
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=191)
    parser.add_argument("--flow-steps", type=int, default=12)
    parser.add_argument("--noise-scale", type=float, default=0.75)
    parser.add_argument("--room-size", type=int, default=224)
    parser.add_argument("--control-source", choices=("aux", "sampled", "blend"), default="blend")
    parser.add_argument("--room-instance-head", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=19)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    model, config, taxonomy, _state = load_checkpoint_model(
        args.checkpoint,
        device,
        taxonomy=None,
    )
    model.eval()
    dataset = CubiCasaFloorplanDataset(
        args.data_root,
        args.split,
        image_size=int(config.get("image_size", 256)),
        cache_root=args.cache_root,
        limit=args.limit,
        augment=False,
        write_cache=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    rows = []
    for _target, footprint, floor_ids, names in loader:
        controls = predict_contour_controls(
            model,
            footprint.to(device),
            taxonomy,
            floor_ids=floor_ids.to(device),
            flow_steps=args.flow_steps,
            noise_scale=args.noise_scale,
            control_source=args.control_source,
            use_room_instance_head=args.room_instance_head,
        )
        room_logits = torch.nn.functional.interpolate(
            controls.room_logits,
            size=(args.room_size, args.room_size),
            mode="bilinear",
            align_corners=False,
        )
        for index, name in enumerate(names):
            path = control_cache_path(args.output_root, args.split, name)
            write_control_cache(
                path,
                room_logits=room_logits[index : index + 1],
                program=controls.program[index : index + 1],
                structure=controls.structure[index : index + 1],
                furniture=controls.furniture[index : index + 1],
                support=controls.support[index : index + 1],
                room_size=args.room_size,
            )
            rows.append(
                {
                    "name": name,
                    "file": str(path),
                    "sha256": sha256_file(path),
                }
            )
        print(f"predicted {len(rows)}/{len(dataset)}", flush=True)
    manifest = {
        "schema_version": CONTROL_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "source": "predicted",
        "analysis_variant": "contour_cfm_guided",
        "target_used_for_control": False,
        "deployable": True,
        "split": args.split,
        "cases": len(rows),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "flow_steps": args.flow_steps,
        "noise_scale": args.noise_scale,
        "control_source": args.control_source,
        "sampling_seed": args.seed,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "taxonomy_sha256": sha256_file(args.cache_root / "taxonomy.json"),
        "control_sizes": {
            "room": args.room_size,
            "structure": controls.structure.shape[-1],
            "furniture": controls.furniture.shape[-1],
            "support": controls.support.shape[-1],
        },
        "output_root": str(args.output_root.resolve()),
        "rows": rows,
    }
    path = args.output_root / args.split / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(path), "cases": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
