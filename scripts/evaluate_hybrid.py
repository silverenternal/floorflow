#!/usr/bin/env python3
"""Evaluate class-group routing between direct-semantic and flow checkpoints."""

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
from floorflow.inference.hybrid_decode import hybrid_decode
from floorflow.evaluation.metrics import floorplan_metrics
from floorflow.model.backbone import FloorplanFlowModel
from floorflow.data.taxonomy import FloorplanTaxonomy


def _load(path: Path, taxonomy: FloorplanTaxonomy, device: str):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["model"]
    config = checkpoint.get("config", {})
    model = FloorplanFlowModel(
        target_channels=int(state["output.weight"].shape[0]),
        base_channels=int(config.get("base_channels", 48)),
        levels=int(config.get("levels", 3)),
        condition_dropout=0.0,
        semantic_channel_attention=any("semantic_channel_attention." in key for key in state),
        semantic_relation_bias=any("semantic_channel_attention.channel_prior" in key for key in state),
        slot_mixing=any("slot_mixer." in key for key in state),
        dynamic_output_head=any("dynamic_output_head." in key for key in state),
        topology_aux=any("topology_aux." in key for key in state),
        floor_conditioning=any("floor_embedding.weight" in key for key in state),
        semantic_endpoint_head=any("semantic_endpoint." in key for key in state),
        num_rooms=len(taxonomy.room_ids),
        channel_relations=taxonomy.channel_prior(),
        advanced_streams=any("layout_stream_mixer." in key for key in state),
    ).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", type=Path, required=True, help="direct semantic checkpoint")
    parser.add_argument("--secondary", type=Path, required=True, help="flow checkpoint")
    parser.add_argument("--cache-root", type=Path, default=REPO_ROOT / "outputs/cache/cubicasa5k_full_v1")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / ".data/cubicasa5k")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs/eval/hybrid/report.json")
    args = parser.parse_args()
    checkpoint = torch.load(args.primary, map_location="cpu", weights_only=False)
    taxonomy = FloorplanTaxonomy.from_json(checkpoint["taxonomy"])
    primary, primary_config = _load(args.primary, taxonomy, args.device)
    secondary, secondary_config = _load(args.secondary, taxonomy, args.device)
    dataset = CubiCasaFloorplanDataset(
        args.data_root, args.split, image_size=int(primary_config.get("image_size", 256)), cache_root=args.cache_root, limit=args.limit
    )
    rows = []
    with torch.no_grad():
        for index in range(len(dataset)):
            target, footprint, floor_id, name = dataset[index]
            primary_x, _ = sample_floorplans(
                primary, footprint[None].to(args.device), steps=args.steps,
                floor_ids=torch.tensor([floor_id], device=args.device),
                objective_mode=primary_config.get("objective_mode", "velocity"), semantic_decode_weight=1.0,
            )
            secondary_x, _ = sample_floorplans(
                secondary, footprint[None].to(args.device), steps=args.steps,
                floor_ids=torch.tensor([floor_id], device=args.device),
                objective_mode=secondary_config.get("objective_mode", "velocity"), semantic_decode_weight=0.0,
            )
            primary_labels = compose_known_exterior(
                decode_floorplans(primary_x),
                footprint[None].to(args.device),
                opening_ids=taxonomy.opening_ids,
            )[0].cpu().numpy()
            secondary_labels = compose_known_exterior(
                decode_floorplans(secondary_x),
                footprint[None].to(args.device),
                opening_ids=taxonomy.opening_ids,
            )[0].cpu().numpy()
            hybrid = hybrid_decode(primary_labels, secondary_labels, taxonomy, secondary_classes=("furniture",))
            metrics = floorplan_metrics(
                torch.from_numpy(hybrid)[None].to(args.device),
                target[None].to(args.device),
                footprint[None].to(args.device),
                taxonomy,
            )
            rows.append({"case": name, **metrics})
    keys = ("room_macro_iou", "furniture_iou", "opening_f1", "interior_wall_iou", "room_area_coverage")
    summary = {f"{key}_mean": float(np.mean([row[key] for row in rows])) for key in keys}
    report = {"protocol": {"steps": args.steps, "limit": len(rows)}, "summary": summary, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
