#!/usr/bin/env python3
"""Inspect which receptive fields the dynamic output head selects per class."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch

from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.flow.core import interpolation, sample_prior, target_to_continuous
from floorflow.model.backbone import FloorplanFlowModel
from floorflow.data.taxonomy import FloorplanTaxonomy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / ".data/cubicasa5k")
    parser.add_argument("--cache-root", type=Path, default=REPO_ROOT / "outputs/cache/cubicasa5k_v2")
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--t", type=float, default=0.8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    taxonomy = (
        FloorplanTaxonomy.from_json(checkpoint["taxonomy"])
        if checkpoint.get("taxonomy")
        else FloorplanTaxonomy.read(args.cache_root / "taxonomy.json")
    )
    state = checkpoint["model"]
    model = FloorplanFlowModel(
        target_channels=int(state["output.weight"].shape[0]),
        base_channels=int(config.get("base_channels", 48)),
        levels=int(config.get("levels", 3)),
        condition_dropout=0.0,
        semantic_channel_attention=any("semantic_channel_attention." in key for key in state),
        semantic_relation_bias=any("semantic_channel_attention.channel_prior" in key for key in state),
        slot_mixing=any("slot_mixer." in key for key in state),
        num_slots=int(config.get("num_slots", 16)),
        dynamic_output_head=any("dynamic_output_head." in key for key in state),
        output_kernels=tuple(int(value) for value in config.get("output_kernels", (3, 5, 7))),
        floor_conditioning=any("floor_embedding.weight" in key for key in state),
        num_floors=int(config.get("num_floors", 4)),
        topology_aux=any("topology_aux." in key for key in state),
        num_rooms=len(taxonomy.room_ids),
        semantic_endpoint_head=any("semantic_endpoint." in key for key in state),
        channel_relations=taxonomy.channel_prior(),
        advanced_streams=any("layout_stream_mixer." in key or "spatial_velocity." in key for key in state),
        fourier_conditioning=int(state["condition_encoder.stem.weight"].shape[1]) >= 10,
        room_partition_head=any("room_partition." in key for key in state),
        num_room_classes=len(taxonomy.room_ids),
        room_partition_semantic_prior="room_partition.prior_logits.weight" in state,
        room_class_ids=tuple(taxonomy.room_ids),
        room_program_head=any("room_program." in key for key in state),
        room_group_head=any("room_group." in key for key in state),
        layoutbridge_refinement=any("dual_stream_injection." in key for key in state),
        layoutbridge_hidden=int(config.get("layoutbridge_hidden", 64)),
        room_mask_head=any("room_mask_decoder." in key for key in state),
        room_mask_queries=int(config.get("room_mask_queries", 32)),
        room_mask_resolution=int(config.get("room_mask_resolution", 64)),
        room_mask_use_partition=(
            "room_mask_decoder.mask_features.weight" in state
            and int(state["room_mask_decoder.mask_features.weight"].shape[1])
            > int(config.get("base_channels", 48))
        ),
        structured_object_head=any("structured_objects." in key for key in state),
        semantic_taxonomy=taxonomy,
        semantic_anchor_fusion=any("semantic_anchor_fusion." in key for key in state),
    ).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    if not model.use_dynamic_output_head:
        raise SystemExit("checkpoint does not contain a dynamic output head")

    dataset = CubiCasaFloorplanDataset(
        args.data_root,
        args.split,
        image_size=int(config.get("image_size", 256)),
        cache_root=args.cache_root,
        limit=args.limit,
    )
    kernel_sizes = model.dynamic_output_head.kernel_sizes
    numerator = torch.zeros(len(kernel_sizes), taxonomy.num_classes, dtype=torch.float64)
    denominator = torch.zeros(taxonomy.num_classes, dtype=torch.float64)
    for index in range(len(dataset)):
        target, footprint, floor_id, _name = dataset[index]
        target = target[None].to(device)
        footprint = footprint[None].to(device)
        x1 = target_to_continuous(target, model.target_channels)
        x0 = sample_prior(footprint, model.target_channels)
        x_t = interpolation(x0, x1, torch.full((1,), float(args.t), device=device))
        model(
            x_t,
            torch.full((1,), float(args.t), device=device),
            footprint,
            floor_ids=torch.tensor([floor_id], device=device),
        )
        weights = model.dynamic_output_head.last_scale_weights[0]
        for class_id in range(taxonomy.num_classes):
            mask = (target[0] == class_id) & (footprint[0] > 0.5)
            if not mask.any():
                continue
            if class_id == 0:
                class_weights = weights.mean(dim=1)
            else:
                class_weights = weights[:, class_id - 1]
            numerator[:, class_id] += class_weights[:, mask].mean(dim=1).cpu().double()
            denominator[class_id] += 1

    rows = []
    for class_id, name in enumerate(taxonomy.class_names):
        if denominator[class_id] == 0:
            continue
        mean_weights = numerator[:, class_id] / denominator[class_id]
        rows.append(
            {
                "class_id": class_id,
                "class_name": name,
                **{f"kernel_{size}": float(mean_weights[idx]) for idx, size in enumerate(kernel_sizes)},
                "preferred_kernel": int(kernel_sizes[int(mean_weights.argmax())]),
            }
        )
    report = {
        "checkpoint": str(args.checkpoint),
        "t": args.t,
        "kernel_sizes": list(kernel_sizes),
        "classes": rows,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
