#!/usr/bin/env python3
"""Evaluate dense rooms plus specialized object-family semantic heads."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from floorflow.training.checkpoint import load_checkpoint_model
from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.geometry.core import compose_known_exterior
from floorflow.inference.core import semantic_endpoint_labels
from floorflow.geometry.interior_wall import InteriorWallHead
from floorflow.evaluation.metrics import floorplan_metrics
from floorflow.inference.retrieval_prior import ContourRoomRetriever, prior_one_hot
from floorflow.heads.room_partition import RoomPartitionHead
from floorflow.inference.room_repair import apply_floorplan_repairs, drop_outside_furniture
from floorflow.inference.sampling_policy import SamplingPolicy
from floorflow.heads.structured_objects import (
    StructuredObjectHeads,
    compatible_furniture_mask,
    compose_structured_objects,
    object_group_predictions,
    partition_wall_mask,
)
from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.visualize import colorize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition-checkpoint", type=Path, required=True)
    parser.add_argument("--object-checkpoint", type=Path, required=True)
    parser.add_argument("--semantic-checkpoint", type=Path, required=True)
    parser.add_argument("--furniture-checkpoint", type=Path, default=None)
    parser.add_argument("--opening-checkpoint", type=Path, default=None)
    parser.add_argument("--wall-checkpoint", type=Path, default=None)
    parser.add_argument("--cache-root", type=Path, default=REPO_ROOT / "outputs/cache/cubicasa5k_full_v1")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / ".data/cubicasa5k")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--opening-threshold", type=float, default=0.0)
    parser.add_argument("--furniture-threshold", type=float, default=0.0)
    parser.add_argument("--wall-threshold", type=float, default=0.0)
    parser.add_argument("--guard-repairs", action="store_true")
    parser.add_argument("--drop-outside-furniture", action="store_true")
    parser.add_argument("--partition-walls", action="store_true")
    parser.add_argument("--partition-wall-radius", type=int, default=1)
    parser.add_argument(
        "--furniture-source",
        choices=(
            "model",
            "prior",
            "union",
            "compatible_union",
            "semantic_compatible_union",
            "selective_prior",
            "selective_prior_agreement",
            "intersection",
        ),
        default="model",
        help="fixed class-routing rule, never candidate selection",
    )
    parser.add_argument("--prior-similarity-threshold", type=float, default=0.99)
    parser.add_argument("--prior-room-agreement-threshold", type=float, default=0.20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument("--save-examples", type=int, default=0)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs/eval/structured/full_test.json")
    args = parser.parse_args()
    partition_payload = torch.load(args.partition_checkpoint, map_location="cpu", weights_only=False)
    object_payload = torch.load(args.object_checkpoint, map_location="cpu", weights_only=False)
    taxonomy = FloorplanTaxonomy.from_json(partition_payload["taxonomy"])
    backbone, config, loaded_taxonomy, _ = load_checkpoint_model(
        args.semantic_checkpoint,
        args.device,
        taxonomy=taxonomy,
    )
    if loaded_taxonomy.class_names != taxonomy.class_names:
        raise ValueError("partition and semantic taxonomy mismatch")
    partition_config = partition_payload.get("config", {})
    retrieval_top_k = int(partition_config.get("retrieval_top_k", 0))
    retriever = None
    if retrieval_top_k > 0 or args.furniture_source != "model":
        train = CubiCasaFloorplanDataset(
            args.data_root,
            "train",
            image_size=int(config.get("image_size", 256)),
            cache_root=args.cache_root,
            preload=True,
        )
        retriever = ContourRoomRetriever(
            train._targets,
            train._footprints,
            taxonomy,
            max(1, retrieval_top_k),
        )
    room_head = RoomPartitionHead(
        backbone.output.in_channels,
        len(taxonomy.room_ids),
        prior_channels=len(taxonomy.room_ids) if retriever is not None else 0,
    ).to(args.device)
    room_head.load_state_dict(partition_payload["head"], strict=False)
    object_head = StructuredObjectHeads(backbone.output.in_channels, taxonomy).to(args.device)
    object_head.load_state_dict(object_payload["head"])
    furniture_model = None
    opening_model = None
    wall_head = None
    if args.furniture_checkpoint is not None:
        furniture_model, _furniture_config, furniture_taxonomy, _ = load_checkpoint_model(
            args.furniture_checkpoint,
            args.device,
            taxonomy=taxonomy,
        )
        if furniture_taxonomy.class_names != taxonomy.class_names:
            raise ValueError("furniture model taxonomy mismatch")
    if args.opening_checkpoint is not None:
        opening_model, _opening_config, opening_taxonomy, _ = load_checkpoint_model(
            args.opening_checkpoint,
            args.device,
            taxonomy=taxonomy,
        )
        if opening_taxonomy.class_names != taxonomy.class_names:
            raise ValueError("opening model taxonomy mismatch")
    if args.wall_checkpoint is not None:
        wall_payload = torch.load(args.wall_checkpoint, map_location="cpu", weights_only=False)
        wall_head = InteriorWallHead(backbone.output.in_channels).to(args.device)
        wall_head.load_state_dict(wall_payload["head"])
        wall_head.eval()
    room_head.eval()
    object_head.eval()
    dataset = CubiCasaFloorplanDataset(
        args.data_root,
        args.split,
        image_size=int(config.get("image_size", 256)),
        cache_root=args.cache_root,
        limit=args.limit,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    room_ids = torch.tensor(taxonomy.room_ids, device=args.device)
    rows = []
    saved = 0
    with torch.no_grad():
        for target, footprint, floor_ids, names in loader:
            target, footprint, floor_ids = target.to(args.device), footprint.to(args.device), floor_ids.to(args.device)
            zero = torch.zeros(target.shape[0], backbone.target_channels, *target.shape[-2:], device=args.device)
            features = backbone._encode_features(
                zero,
                torch.zeros(target.shape[0], device=args.device),
                footprint,
                floor_ids=floor_ids,
            )
            prior = None
            if retriever is not None:
                prior = prior_one_hot(
                    retriever.retrieve(footprint.cpu().numpy()),
                    len(taxonomy.room_ids),
                    args.device,
                )
            rooms = room_ids[room_head(features, prior).argmax(dim=1)]
            objects = object_group_predictions(
                object_head(features),
                taxonomy,
                opening_threshold=args.opening_threshold,
                furniture_threshold=args.furniture_threshold,
                wall_threshold=args.wall_threshold,
            )
            if furniture_model is not None and args.furniture_source == "model":
                external = semantic_endpoint_labels(furniture_model, target, footprint, floor_ids)
                objects["furniture"] = torch.where(
                    torch.isin(external, torch.tensor(taxonomy.furniture_ids, device=args.device)),
                    external,
                    torch.zeros_like(external),
                )
            if opening_model is not None:
                external = semantic_endpoint_labels(opening_model, target, footprint, floor_ids)
                objects["opening"] = torch.where(
                    torch.isin(external, torch.tensor(taxonomy.opening_ids, device=args.device)),
                    external,
                    torch.zeros_like(external),
                )
            if wall_head is not None:
                objects["wall"] = (wall_head(features) > args.wall_threshold).long()
            if args.partition_walls:
                objects["wall"] = objects["wall"] | partition_wall_mask(
                    rooms,
                    radius=max(0, args.partition_wall_radius),
                ).long()
            if retriever is not None:
                nearest_labels = torch.from_numpy(
                    retriever.retrieve_labels(footprint.cpu().numpy())
                ).to(args.device)
                prior_furniture = torch.where(
                    torch.isin(nearest_labels, torch.tensor(taxonomy.furniture_ids, device=args.device)),
                    nearest_labels,
                    torch.zeros_like(nearest_labels),
                )
                model_furniture = objects["furniture"]
                if args.furniture_source == "prior":
                    objects["furniture"] = prior_furniture
                elif args.furniture_source == "union":
                    objects["furniture"] = torch.where(
                        model_furniture != 0,
                        model_furniture,
                        prior_furniture,
                    )
                elif args.furniture_source == "compatible_union":
                    predicted_rooms = torch.isin(
                        rooms,
                        torch.tensor(taxonomy.room_ids, device=args.device),
                    ).float()[:, None]
                    room_neighborhood = F.max_pool2d(predicted_rooms, 5, stride=1, padding=2)[:, 0] > 0.5
                    compatible_prior = torch.where(
                        room_neighborhood,
                        prior_furniture,
                        torch.zeros_like(prior_furniture),
                    )
                    objects["furniture"] = torch.where(
                        model_furniture != 0,
                        model_furniture,
                        compatible_prior,
                    )
                elif args.furniture_source == "semantic_compatible_union":
                    compatible_prior = torch.where(
                        compatible_furniture_mask(prior_furniture, rooms, taxonomy),
                        prior_furniture,
                        torch.zeros_like(prior_furniture),
                    )
                    objects["furniture"] = torch.where(
                        model_furniture != 0,
                        model_furniture,
                        compatible_prior,
                    )
                elif args.furniture_source == "selective_prior":
                    similarity = torch.from_numpy(
                        retriever.nearest_similarity(footprint.cpu().numpy())
                    ).to(args.device, dtype=torch.float32)
                    use_prior = (similarity >= args.prior_similarity_threshold)[:, None, None]
                    objects["furniture"] = torch.where(
                        use_prior,
                        prior_furniture,
                        model_furniture,
                    )
                elif args.furniture_source == "selective_prior_agreement":
                    union = (rooms != 0) | torch.isin(
                        nearest_labels,
                        torch.tensor(taxonomy.room_ids, device=args.device),
                    )
                    agreement = (
                        ((rooms == nearest_labels) & union & (footprint > 0.5)).sum(dim=(1, 2)).float()
                        / union.sum(dim=(1, 2)).clamp_min(1).float()
                    )
                    use_prior = (agreement >= args.prior_room_agreement_threshold)[:, None, None]
                    objects["furniture"] = torch.where(
                        use_prior,
                        prior_furniture,
                        model_furniture,
                    )
                elif args.furniture_source == "intersection":
                    objects["furniture"] = torch.where(
                        (model_furniture != 0) & (prior_furniture != 0),
                        model_furniture,
                        torch.zeros_like(model_furniture),
                    )
            labels = compose_structured_objects(rooms, objects, footprint, taxonomy)
            if args.guard_repairs:
                labels = torch.stack(
                    [
                        torch.from_numpy(
                            apply_floorplan_repairs(
                                labels[index].cpu().numpy(),
                                footprint[index].cpu().numpy().astype("uint8"),
                                taxonomy,
                            )
                        ).to(args.device)
                        for index in range(labels.shape[0])
                    ]
                )
            elif args.drop_outside_furniture:
                labels = torch.stack(
                    [
                        torch.from_numpy(
                            drop_outside_furniture(
                                labels[index].cpu().numpy(),
                                taxonomy,
                            )
                        ).to(args.device)
                        for index in range(labels.shape[0])
                    ]
                )
            labels = compose_known_exterior(labels, footprint, opening_ids=taxonomy.opening_ids)
            for index, name in enumerate(names):
                if args.image_dir is not None and saved < args.save_examples:
                    args.image_dir.mkdir(parents=True, exist_ok=True)
                    stem = name.replace("/", "__").replace("\\", "__")
                    Image.fromarray(colorize(labels[index].cpu().numpy(), taxonomy)).save(
                        args.image_dir / f"{stem}.png"
                    )
                    saved += 1
                rows.append(
                    {
                        "case": name,
                        **floorplan_metrics(
                            labels[index : index + 1],
                            target[index : index + 1],
                            footprint[index : index + 1],
                            taxonomy,
                        ),
                    }
                )
    keys = (
        "room_macro_iou",
        "room_area_coverage",
        "room_component_gap",
        "room_area_distribution_mae",
        "interior_wall_iou",
        "wall_iou",
        "wall_closure",
        "wall_centerline_f1",
        "opening_f1",
        "opening_wall_alignment",
        "furniture_iou",
        "furniture_room_containment",
        "outside_footprint_fraction",
        "exterior_band_is_known_wall",
        "present_class_macro_iou",
    )
    summary = {key: float(np.mean([row[key] for row in rows])) for key in keys}
    report = {
        "cases": len(rows),
        "claim_role": "dense_partition_and_specialized_objects",
        "best_of_n_used": False,
        "sampling_policy": SamplingPolicy(purpose="single_sample").manifest(),
        "thresholds": {
            "opening": args.opening_threshold,
            "furniture": args.furniture_threshold,
            "wall": args.wall_threshold,
        },
        "furniture_source": args.furniture_source,
        "guard_repairs": args.guard_repairs,
        "drop_outside_furniture": args.drop_outside_furniture,
        "partition_walls": args.partition_walls,
        "partition_wall_radius": args.partition_wall_radius,
        "furniture_checkpoint": str(args.furniture_checkpoint) if args.furniture_checkpoint is not None else None,
        "opening_checkpoint": str(args.opening_checkpoint) if args.opening_checkpoint is not None else None,
        "wall_checkpoint": str(args.wall_checkpoint) if args.wall_checkpoint is not None else None,
        "prior_similarity_threshold": args.prior_similarity_threshold,
        "prior_room_agreement_threshold": args.prior_room_agreement_threshold,
        "metrics": summary,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"cases": len(rows), "metrics": summary}, indent=2))


if __name__ == "__main__":
    main()
