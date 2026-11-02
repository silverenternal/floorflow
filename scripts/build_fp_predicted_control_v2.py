#!/usr/bin/env python3
"""Build target-free predicted controls from the retained FP model family."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from floorflow.data.control_cache import control_cache_path, write_control_cache
from floorflow.data.control_contract import CONTROL_SCHEMA_VERSION, sha256_file
from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.heads.coarse_furniture_refiner import (
    CoarseFurnitureRefiner,
    coarse_furniture_control,
)
from floorflow.heads.coarse_structure_refiner import (
    CoarseStructureRefiner,
    coarse_structure_control,
)
from floorflow.heads.coarse_support_refiner import coarse_support_control
from floorflow.heads.room_program import gate_room_labels_to_program
from floorflow.heads.structured_objects import partition_wall_mask
from floorflow.inference.contour_program import (
    load_contour_layout_predictor,
    load_room_program_predictor,
    predict_room_program,
)
from floorflow.representation.contour_memory import ContourPartitionMemory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path(".data/cubicasa5k"))
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("outputs/cache/cubicasa5k_full_v2"),
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=191)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--program-checkpoint",
        type=Path,
        default=Path("outputs/train/room_program_set_full_v3/best.pt"),
    )
    parser.add_argument(
        "--layout-checkpoint",
        type=Path,
        default=Path("outputs/train/contour_program_layout_full_v1/best.pt"),
    )
    parser.add_argument(
        "--structure-checkpoint",
        type=Path,
        default=Path("outputs/train/coarse_structure_type_full_v1/best.pt"),
    )
    parser.add_argument(
        "--furniture-checkpoint",
        type=Path,
        default=Path("outputs/train/coarse_furniture_full_s192_v1/best.pt"),
    )
    parser.add_argument("--room-size", type=int, default=224)
    parser.add_argument("--structure-size", type=int, default=128)
    parser.add_argument("--furniture-size", type=int, default=192)
    parser.add_argument("--support-size", type=int, default=192)
    parser.add_argument(
        "--retrieval-weight",
        type=float,
        default=0.0,
        help="Blend the contour-layout prediction with train-only retrieval.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _load_structure_model(path: Path, taxonomy, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = CoarseStructureRefiner(
        len(taxonomy.room_ids) + len(taxonomy.opening_ids) + 3,
        len(taxonomy.opening_ids),
        base_channels=int(payload["config"].get("base_channels", 32)),
    ).to(device)
    model.load_state_dict(payload["model"])
    return model.eval()


def _load_furniture_model(path: Path, taxonomy, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = CoarseFurnitureRefiner(
        len(taxonomy.room_ids) + len(taxonomy.furniture_ids) + 1,
        len(taxonomy.furniture_ids),
        base_channels=int(payload["config"].get("base_channels", 32)),
    ).to(device)
    model.load_state_dict(payload["model"])
    return model.eval()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dataset = CubiCasaFloorplanDataset(
        args.data_root,
        args.split,
        image_size=256,
        cache_root=args.cache_root,
        limit=args.limit,
        augment=False,
        write_cache=False,
    )
    taxonomy = dataset.taxonomy
    room_ids = torch.tensor(taxonomy.room_ids, device=device)
    backbone, program_model = load_room_program_predictor(
        args.program_checkpoint,
        taxonomy,
        device,
    )
    layout_model = load_contour_layout_predictor(
        args.layout_checkpoint,
        len(taxonomy.room_ids),
        device,
    )
    structure_model = _load_structure_model(
        args.structure_checkpoint,
        taxonomy,
        device,
    )
    furniture_model = _load_furniture_model(
        args.furniture_checkpoint,
        taxonomy,
        device,
    )
    memory = None
    if args.retrieval_weight > 0.0:
        memory = ContourPartitionMemory.from_cache(
            args.cache_root,
            top_k=1,
            metric="cosine",
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    rows = []
    for _target, footprint, floor_ids, names in loader:
        footprint = footprint.to(device)
        floor_ids = floor_ids.to(device)
        program = predict_room_program(
            backbone,
            program_model,
            footprint,
            floor_ids,
        )
        layout_logits = layout_model(footprint, program.float())
        if memory is not None:
            retrieval = memory.retrieve_logits(
                footprint,
                taxonomy,
                device,
            )
            layout_logits = (
                (1.0 - args.retrieval_weight) * layout_logits
                + args.retrieval_weight * retrieval
            )
        rooms = room_ids[layout_logits.argmax(dim=1)]
        rooms = gate_room_labels_to_program(
            rooms,
            program,
            taxonomy.room_ids,
        ) * footprint.long()
        room_control = F.interpolate(
            layout_logits,
            size=(args.room_size, args.room_size),
            mode="bilinear",
            align_corners=False,
        )
        structure_input = coarse_structure_control(
            rooms,
            footprint,
            taxonomy,
            size=args.structure_size,
        )
        structure = structure_model(
            structure_input,
            output_size=(args.structure_size, args.structure_size),
        )
        room_count = len(taxonomy.room_ids)
        structure_control = structure_input.clone()
        boundary_wall = partition_wall_mask(rooms, radius=2)
        boundary_wall = F.interpolate(
            boundary_wall.float()[:, None],
            size=(args.structure_size, args.structure_size),
            mode="nearest",
        )[:, 0]
        structure_control[:, room_count] = torch.maximum(
            structure["wall_logits"].sigmoid()[:, 0],
            boundary_wall,
        )
        structure_control[:, room_count + 1] = structure[
            "opening_logits"
        ].sigmoid()[:, 0]
        structure_control[
            :,
            room_count + 2 : room_count + 2 + len(taxonomy.opening_ids),
        ] = structure["opening_type_logits"].softmax(dim=1)
        furniture_input = coarse_furniture_control(
            rooms,
            footprint,
            taxonomy,
            size=args.furniture_size,
        )
        furniture = furniture_model(
            furniture_input,
            output_size=(args.furniture_size, args.furniture_size),
        )
        furniture_control = furniture_input.clone()
        furniture_control[:, room_count:-1] = (
            furniture["class_logits"].softmax(dim=1)
            * furniture["foreground_logits"].sigmoid()
        )
        support_control = coarse_support_control(
            rooms,
            footprint,
            taxonomy,
            size=args.support_size,
        )
        for index, name in enumerate(names):
            path = control_cache_path(args.output_root, args.split, name)
            write_control_cache(
                path,
                room_logits=room_control[index : index + 1],
                program=program[index : index + 1],
                structure=structure_control[index : index + 1],
                furniture=furniture_control[index : index + 1],
                support=support_control[index : index + 1],
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
        "analysis_variant": "contour_program_layout_v2_stage1",
        "target_used_for_control": False,
        "deployable": True,
        "split": args.split,
        "cases": len(rows),
        "program_checkpoint": str(args.program_checkpoint.resolve()),
        "program_checkpoint_sha256": sha256_file(args.program_checkpoint),
        "layout_checkpoint": str(args.layout_checkpoint.resolve()),
        "layout_checkpoint_sha256": sha256_file(args.layout_checkpoint),
        "structure_checkpoint": str(args.structure_checkpoint.resolve()),
        "structure_checkpoint_sha256": sha256_file(args.structure_checkpoint),
        "furniture_checkpoint": str(args.furniture_checkpoint.resolve()),
        "furniture_checkpoint_sha256": sha256_file(args.furniture_checkpoint),
        "retrieval_weight": args.retrieval_weight,
        "control_sizes": {
            "room": args.room_size,
            "structure": args.structure_size,
            "furniture": args.furniture_size,
            "support": args.support_size,
        },
        "taxonomy_sha256": sha256_file(args.cache_root / "taxonomy.json"),
        "output_root": str(args.output_root.resolve()),
        "rows": rows,
    }
    path = args.output_root / args.split / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(path), "cases": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
