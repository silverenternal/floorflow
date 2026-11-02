#!/usr/bin/env python3
"""Evaluate a learned vector scaffold as a composed CAD plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_raster_predicted_control_bridge import RasterPredictedControlDataset
from train_residential_predicted_refiner import _predicted_controls
from train_vector_topology_scaffold_v1 import GRAPH_STAGES

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.fpeval_bridge import evaluate, to_fpeval_plan
from floorflow.geometry.wall_graph_readout import raster_labels_to_wall_plan
from floorflow.heads.vector_primitive_head import VectorPrimitiveHead
from floorflow.inference.cad_stage_runtime import CadStageRuntime
from floorflow.inference.vector_scaffold_readout import vector_scaffold_plan
from floorflow.model.predicted_channel_checkpoint import (
    build_predicted_channel_model,
)

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer-checkpoint", type=Path, required=True)
    parser.add_argument("--scaffold-checkpoint", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    layer_payload = torch.load(
        args.layer_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    scaffold_payload = torch.load(
        args.scaffold_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    config = layer_payload["config"]
    scaffold_config = scaffold_payload["config"]
    taxonomy = FloorplanTaxonomy.from_json(layer_payload["taxonomy"])
    layer_model = build_predicted_channel_model(taxonomy, config).to(device)
    layer_model.load_state_dict(layer_payload["model"])
    layer_model.eval()
    scaffold = VectorPrimitiveHead(
        int(config["field_channels"]),
        len(taxonomy.room_ids),
        len(taxonomy.opening_ids),
        hidden_dim=int(scaffold_config["hidden_dim"]),
        max_rooms=int(scaffold_config.get("room_queries", 4)),
        max_walls=int(scaffold_config["max_walls"]),
        max_openings=int(scaffold_config["max_openings"]),
        polygon_points=8,
        decoder_layers=2,
        attention_heads=4,
        stage_context=bool(scaffold_config.get("stage_context", False)),
        host_conditioned_openings=bool(
            scaffold_config.get("host_conditioned_openings", False)
        ),
    ).to(device)
    scaffold.load_state_dict(scaffold_payload["scaffold"])
    scaffold.eval()
    graph_projection = torch.nn.Linear(
        int(config["field_channels"]),
        int(scaffold_config["hidden_dim"]),
    ).to(device)
    graph_projection.load_state_dict(scaffold_payload["graph_projection"])
    graph_projection.eval()
    stage_runtime = CadStageRuntime(
        taxonomy,
        device,
        room_checkpoint=config["room_checkpoint"],
        structure_checkpoint=config["structure_checkpoint"],
        centerline_checkpoint=config["centerline_checkpoint"],
        opening_checkpoint=config["opening_checkpoint"],
        room_blend=float(config["room_blend"]),
        structure_size=128,
    )
    dataset = RasterPredictedControlDataset(
        Path(config["predicted_root"]),
        "val",
        Path(config["cache_root"]),
        Path(config["data_root"]),
        image_size=int(config["image_size"]),
        limit=args.limit,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    rows = []
    for target, footprint, floor_ids, names, image in loader:
        target = target.to(device)
        footprint = footprint.to(device)
        controls = _predicted_controls(
            Path(config["predicted_root"]),
            "val",
            names,
            type(
                "Args",
                (),
                {
                    "predicted_root": Path(config["predicted_root"]),
                    "room_size": 224,
                    "structure_size": 128,
                },
            )(),
            device,
        )
        controls, _diagnostics = stage_runtime(
            controls,
            footprint,
            image.to(device),
            floor_ids.to(device),
        )
        layer_output = layer_model(controls, footprint)
        tokens = torch.stack(
            [layer_output["action_tokens"][name] for name in GRAPH_STAGES],
            dim=1,
        )
        vectors = scaffold(
            layer_output["field_features"],
            graph_tokens=graph_projection(tokens),
            graph_mask=torch.ones(
                tokens.shape[:2],
                dtype=torch.bool,
                device=device,
            ),
        )
        room_logits = torch.nn.functional.interpolate(
            layer_output["cad_layers"].room_logits,
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        for index, name in enumerate(names):
            plan = vector_scaffold_plan(
                room_logits[index],
                footprint[index],
                vectors["wall_endpoints"][index],
                vectors["wall_exists"][index],
                vectors["wall_thickness"][index],
                vectors["opening_centers"][index],
                vectors["opening_exists"][index],
                vectors["opening_widths"][index],
                vectors["opening_class"][index],
                taxonomy,
                threshold=args.threshold,
                opening_host_logits=vectors.get("opening_host_logits", None)[index]
                if "opening_host_logits" in vectors
                else None,
            )
            target_plan = to_fpeval_plan(
                raster_labels_to_wall_plan(
                    target[index].cpu().numpy(),
                    taxonomy,
                )
            )
            metrics = evaluate(
                to_fpeval_plan(plan),
                target_plan,
            )
            rows.append(
                {
                    "case": str(name),
                    "wall_f1": float(metrics.walls["f1"]),
                    "opening_f1": float(metrics.openings["f1"]),
                    "room_f1": float(metrics.rooms["f1"]),
                    "validity": float(metrics.r_val),
                    "watertight": float(metrics.r_val),
                    "walls": len(plan.walls),
                    "openings": len(plan.openings),
                    "rooms": len(plan.rooms),
                }
            )
    summary = {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]
        if key != "case"
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "schema_version": "vector-topology-scaffold-eval-v1",
                "checkpoint": str(args.scaffold_checkpoint),
                "test_opened": False,
                "target_used_at_inference": False,
                "summary": summary,
                "cases": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
