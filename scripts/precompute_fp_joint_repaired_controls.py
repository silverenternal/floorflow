#!/usr/bin/env python3
"""Apply joint room-structure repair to a corrupted control contract."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from floorflow.data.control_cache import (
    CachedValidityDataset,
    control_cache_path,
    write_control_cache,
)
from floorflow.data.control_contract import (
    CONTROL_SCHEMA_VERSION,
    sha256_file,
    summarize_control_file,
)
from floorflow.data.control_fusion import inpaint_controls, nearest_fill_batch
from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.heads.joint_control_loading import load_joint_control_repair


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path(".data/cubicasa5k"))
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("outputs/cache/cubicasa5k_full_v2"),
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--joint-weight",
        type=float,
        default=1.0,
        help="Blend learned repair with nearest inpaint inside missing room regions.",
    )
    return parser.parse_args()


def _preservation(
    supplied: torch.Tensor,
    repaired: torch.Tensor,
    valid: torch.Tensor,
) -> float:
    valid = valid.bool()
    if not valid.any():
        return 1.0
    equal = supplied[..., valid] == repaired[..., valid]
    return float(equal.float().mean())


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    base = CubiCasaFloorplanDataset(
        args.data_root,
        args.split,
        image_size=256,
        cache_root=args.cache_root,
        limit=args.limit,
        augment=False,
        write_cache=False,
    )
    taxonomy = base.taxonomy
    model, _config = load_joint_control_repair(
        args.checkpoint,
        taxonomy,
        device,
    )
    dataset = CachedValidityDataset(base, args.control_root, args.split)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    source_manifest_path = args.control_root / args.split / "manifest.json"
    source_manifest = json.loads(
        source_manifest_path.read_text(encoding="utf-8")
    )
    rows = []
    room_preservation = []
    structure_preservation = []
    for target, footprint, _floor_ids, names, controls, masks in loader:
        del target
        footprint = footprint.to(device)
        controls = {key: value.to(device) for key, value in controls.items()}
        masks_device = {
            key: value.to(device).bool() for key, value in masks.items()
        }
        room_input = nearest_fill_batch(
            controls["room_logits"],
            masks_device["valid_room_logits"],
        )
        structure_input = nearest_fill_batch(
            controls["structure"],
            masks_device["valid_structure"],
        )
        output = model(
            room_input,
            structure_input,
            footprint,
            masks_device["valid_room_logits"],
            masks_device["valid_structure"],
        )
        masks_numpy = {
            key: value.cpu().numpy() for key, value in masks.items()
        }
        for index, name in enumerate(names):
            nearest = inpaint_controls(
                {
                    key: value[index]
                    for key, value in controls.items()
                },
                {
                    key: value[index]
                    for key, value in masks_numpy.items()
                },
            )
            room_valid = masks_device["valid_room_logits"][index]
            structure_valid = masks_device["valid_structure"][index]
            repaired_room = output["room_logits"][index]
            if args.joint_weight < 1.0:
                blended_missing = (
                    args.joint_weight * repaired_room
                    + (1.0 - args.joint_weight) * room_input[index]
                )
                repaired_room = torch.where(
                    room_valid[None],
                    controls["room_logits"][index],
                    blended_missing,
                )
            room_preservation.append(
                _preservation(
                    controls["room_logits"][index],
                    repaired_room,
                    room_valid,
                )
            )
            structure_preservation.append(
                _preservation(
                    controls["structure"][index],
                    output["structure_control"][index],
                    structure_valid,
                )
            )
            path = control_cache_path(args.output_root, args.split, name)
            write_control_cache(
                path,
                room_logits=repaired_room[None],
                program=controls["program"][index : index + 1],
                structure=output["structure_control"][index : index + 1],
                furniture=nearest["furniture"][None],
                support=nearest["support"][None],
                room_size=output["room_logits"].shape[-1],
                validity_masks={
                    key: value[index].cpu().numpy()
                    for key, value in masks.items()
                },
            )
            summary = summarize_control_file(
                path,
                taxonomy,
                room_size=output["room_logits"].shape[-1],
                structure_size=output["structure_control"].shape[-1],
                furniture_size=nearest["furniture"].shape[-1],
                support_size=nearest["support"].shape[-1],
            )
            rows.append(
                {
                    "name": name,
                    "file": summary.pop("path"),
                    "sha256": summary.pop("sha256"),
                }
            )
        print(f"repaired {len(rows)}/{len(dataset)}", flush=True)
    manifest = {
        "schema_version": CONTROL_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "source": "oracle",
        "analysis_variant": "joint_room_structure_repair",
        "target_used_for_control": True,
        "deployable": False,
        "split": args.split,
        "cases": len(rows),
        "source_manifest": str(source_manifest_path.resolve()),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "repair_checkpoint": str(args.checkpoint.resolve()),
        "repair_checkpoint_sha256": sha256_file(args.checkpoint),
        "taxonomy_sha256": source_manifest["taxonomy_sha256"],
        "control_sizes": {
            "room": output["room_logits"].shape[-1],
            "structure": output["structure_control"].shape[-1],
            "furniture": nearest["furniture"].shape[-1],
            "support": nearest["support"].shape[-1],
        },
        "fallback_non_target_channels": "nearest_valid_control",
        "room_repair_blend": {
            "learned_weight": args.joint_weight,
            "nearest_weight": 1.0 - args.joint_weight,
        },
        "valid_region_preservation": {
            "room_mean": sum(room_preservation) / len(room_preservation),
            "structure_mean": (
                sum(structure_preservation) / len(structure_preservation)
            ),
            "minimum": min(room_preservation + structure_preservation),
        },
        "output_root": str(args.output_root.resolve()),
        "rows": rows,
    }
    path = args.output_root / args.split / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "manifest": str(path),
                "cases": len(rows),
                "valid_region_preservation": manifest[
                    "valid_region_preservation"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
