#!/usr/bin/env python3
"""Sample floorplans from an exterior boundary and save previews."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch

from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.flow.core import decode_floorplans, sample_floorplans
from floorflow.geometry.core import compose_known_exterior
from floorflow.evaluation.metrics import floorplan_metrics
from floorflow.model.backbone import FloorplanFlowModel
from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.visualize import save_comparison_panel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / ".data/cubicasa5k")
    parser.add_argument("--cache-root", type=Path, default=REPO_ROOT / "outputs/cache/cubicasa5k")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs/samples")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--no-heun", action="store_true")
    parser.add_argument("--noise-scale", type=float, default=0.75)
    parser.add_argument("--smooth-kernel", type=int, default=1)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--objective-mode", choices=("velocity", "simplex"), default=None)
    parser.add_argument("--weight-source", choices=("model", "ema"), default="model")
    parser.add_argument("--semantic-decode-weight", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=11)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    state = checkpoint["ema_model"] if args.weight_source == "ema" and checkpoint.get("ema_model") else checkpoint["model"]
    taxonomy = (
        FloorplanTaxonomy.from_json(checkpoint["taxonomy"])
        if checkpoint.get("taxonomy")
        else FloorplanTaxonomy.read(args.cache_root / "taxonomy.json")
    )
    use_semantic_attention = any("semantic_channel_attention." in key for key in state)
    use_relation_bias = any("semantic_channel_attention.channel_prior" in key for key in state)
    use_slot_mixing = any("slot_mixer." in key for key in state)
    target_channels = int(state["output.weight"].shape[0])
    model = FloorplanFlowModel(
        target_channels=target_channels,
        base_channels=int(config.get("base_channels", 48)),
        levels=int(config.get("levels", 3)),
        condition_dropout=0.0,
        semantic_channel_attention=use_semantic_attention,
        semantic_relation_bias=use_relation_bias,
        slot_mixing=use_slot_mixing,
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
        fourier_conditioning=int(checkpoint["model"]["condition_encoder.stem.weight"].shape[1]) >= 10,
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
    dataset = CubiCasaFloorplanDataset(
        args.data_root,
        args.split,
        image_size=int(config.get("image_size", 256)),
        cache_root=args.cache_root,
        augment=False,
        limit=args.limit,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for index in range(len(dataset)):
        target, footprint, floor_id, name = dataset[index]
        prediction, _trajectory = sample_floorplans(
            model,
            footprint[None].to(device),
            steps=args.steps,
            heun=not args.no_heun,
            noise_scale=args.noise_scale,
            floor_ids=torch.tensor([floor_id], device=device),
            guidance_scale=args.guidance_scale,
            objective_mode=objective_mode,
            semantic_decode_weight=args.semantic_decode_weight,
        )
        labels = compose_known_exterior(
            decode_floorplans(prediction, smooth_kernel=args.smooth_kernel),
            footprint[None].to(device),
            opening_ids=taxonomy.opening_ids,
        )
        prediction_np = labels[0].cpu().numpy()
        target_np = target.numpy()
        footprint_np = footprint.numpy().astype(np.uint8)
        stem = name.replace("/", "__").replace("\\", "__")
        save_comparison_panel(
            args.output_dir / f"{stem}.png",
            footprint_np,
            target_np,
            prediction_np,
            title=name,
            taxonomy=taxonomy,
        )
        summaries.append(
            {
                "case": name,
                "metrics": floorplan_metrics(labels.cpu(), target[None], footprint[None], taxonomy),
            }
        )
    (args.output_dir / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"samples": len(summaries), "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
