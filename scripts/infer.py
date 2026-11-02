#!/usr/bin/env python3
"""Generate floorplan interiors from a standalone exterior-contour image."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch
from PIL import Image

from floorflow.flow.core import decode_floorplans, sample_floorplans
from floorflow.geometry.core import compose_known_exterior
from floorflow.representation.masks import load_footprint
from floorflow.model.backbone import FloorplanFlowModel
from floorflow.inference.room_repair import apply_floorplan_repairs
from floorflow.inference.sampling_policy import SamplingPolicy
from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.visualize import colorize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--footprint", type=Path, required=True, help="exterior contour image")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs/infer")
    parser.add_argument("--foreground", choices=("white", "black"), default="white")
    parser.add_argument("--fill-outline", action="store_true", help="fill the largest enclosed contour region")
    parser.add_argument("--samples", type=int, default=1, help="generate diverse alternatives; no best-of-N selection")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--no-heun", action="store_true")
    parser.add_argument("--noise-scale", type=float, default=0.75)
    parser.add_argument("--smooth-kernel", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--floor-id", type=int, default=1)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--objective-mode", choices=("velocity", "simplex"), default=None)
    parser.add_argument("--weight-source", choices=("model", "ema"), default="model")
    parser.add_argument("--semantic-decode-weight", type=float, default=0.5)
    parser.add_argument("--repair-rooms", action="store_true")
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
        else FloorplanTaxonomy.read(REPO_ROOT / "outputs/cache/cubicasa5k_v2/taxonomy.json")
    )
    state = checkpoint["ema_model"] if args.weight_source == "ema" and checkpoint.get("ema_model") else checkpoint["model"]
    image_size = int(config.get("image_size", 256))
    target_channels = int(state["output.weight"].shape[0])
    model = FloorplanFlowModel(
        target_channels=target_channels,
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
    objective_mode = args.objective_mode or config.get("objective_mode", "velocity")

    footprint_np = load_footprint(
        args.footprint,
        image_size=image_size,
        foreground=args.foreground,
        fill_outline=args.fill_outline,
    )
    footprint = torch.from_numpy(footprint_np)[None].to(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(footprint_np * 255).save(args.output_dir / "footprint.png")
    records = []
    for sample_index in range(max(1, args.samples)):
        torch.manual_seed(args.seed + sample_index)
        sampled, _trajectory = sample_floorplans(
            model,
            footprint,
            steps=args.steps,
            heun=not args.no_heun,
            noise_scale=args.noise_scale,
            floor_ids=torch.tensor([args.floor_id], device=device),
            guidance_scale=args.guidance_scale,
            objective_mode=objective_mode,
            semantic_decode_weight=args.semantic_decode_weight,
        )
        labels = compose_known_exterior(
            decode_floorplans(sampled, smooth_kernel=args.smooth_kernel),
            footprint,
            opening_ids=taxonomy.opening_ids,
        )[0].cpu().numpy()
        if args.repair_rooms:
            labels = apply_floorplan_repairs(labels, footprint_np, taxonomy)
        np.save(args.output_dir / f"sample_{sample_index}.npy", labels)
        Image.fromarray(labels.astype(np.uint8)).save(args.output_dir / f"sample_{sample_index}_labels.png")
        Image.fromarray(colorize(labels, taxonomy)).save(args.output_dir / f"sample_{sample_index}.png")
        records.append({"sample": sample_index, "seed": args.seed + sample_index})
    manifest = {
        "checkpoint": str(args.checkpoint),
        "footprint": str(args.footprint),
        "image_size": image_size,
        "target_channels": target_channels,
        "floor_id": args.floor_id,
        "class_names": taxonomy.class_names,
        "class_colors_rgb": [taxonomy.color(index) for index in range(taxonomy.num_classes)],
        "samples": records,
        "best_of_n_used": False,
        "sampling_policy": SamplingPolicy(
            purpose="single_sample" if args.samples == 1 else "distribution",
            candidates_per_condition=max(1, args.samples),
            seed=args.seed,
        ).manifest(),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "samples": len(records)}, indent=2))


if __name__ == "__main__":
    main()
