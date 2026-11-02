#!/usr/bin/env python3
"""Apply a mask-aware room repair model only inside invalid control regions."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import torch
import torch.nn.functional as F
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
from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.heads.coarse_layout_refiner import CoarseLayoutRefiner


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
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
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
    model = CoarseLayoutRefiner(
        len(taxonomy.room_ids),
        base_channels=int(payload.get("config", {}).get("base_channels", 32)),
        masked_input=True,
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
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
    for target, footprint, _floor_ids, names, controls, masks in loader:
        footprint = footprint.to(device)
        controls = {key: value.to(device) for key, value in controls.items()}
        valid = masks["valid_room_logits"].to(device).bool()
        repaired = model(
            controls["room_logits"],
            footprint,
            valid,
        )
        repaired = F.interpolate(
            repaired,
            size=controls["room_logits"].shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        control_footprint = F.interpolate(
            footprint[:, None],
            size=repaired.shape[-2:],
            mode="nearest",
        )
        fused = torch.where(
            valid[:, None],
            controls["room_logits"],
            repaired,
        ) * control_footprint.bool()
        program = fused.amax(dim=(2, 3)) > 0.1
        missing = ~program.any(dim=1)
        if missing.any():
            fallback = fused.amax(dim=(2, 3)).argmax(dim=1)
            missing_index = torch.nonzero(missing, as_tuple=False).flatten()
            program[missing_index, fallback[missing_index]] = True
        for index, name in enumerate(names):
            path = control_cache_path(args.output_root, args.split, name)
            sample_masks = {
                key: value[index].cpu().numpy() for key, value in masks.items()
            }
            write_control_cache(
                path,
                room_logits=fused[index : index + 1],
                program=program[index : index + 1],
                structure=controls["structure"][index : index + 1],
                furniture=controls["furniture"][index : index + 1],
                support=controls["support"][index : index + 1],
                room_size=fused.shape[-1],
                validity_masks=sample_masks,
            )
            summary = summarize_control_file(
                path,
                taxonomy,
                room_size=fused.shape[-1],
                structure_size=controls["structure"].shape[-1],
                furniture_size=controls["furniture"].shape[-1],
                support_size=controls["support"].shape[-1],
            )
            rows.append(
                {
                    "name": name,
                    "file": summary.pop("path"),
                    **summary,
                }
            )
        print(f"repaired {len(rows)}/{len(dataset)}", flush=True)
    manifest = {
        "schema_version": CONTROL_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "source": "oracle",
        "analysis_variant": "learned_room_repair",
        "target_used_for_control": True,
        "deployable": False,
        "split": args.split,
        "cases": len(rows),
        "source_manifest": str(source_manifest_path.resolve()),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "repair_checkpoint": str(args.checkpoint.resolve()),
        "repair_checkpoint_sha256": sha256_file(args.checkpoint),
        "taxonomy_sha256": source_manifest["taxonomy_sha256"],
        "control_sizes": source_manifest["control_sizes"],
        "output_root": str(args.output_root.resolve()),
        "rows": rows,
    }
    path = args.output_root / args.split / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(path), "cases": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
