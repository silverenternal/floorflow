#!/usr/bin/env python3
"""Evaluate a room-slot furniture vector checkpoint with per-case rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_cad_furniture_vector_v1 import evaluate
from train_raster_predicted_control_bridge import RasterPredictedControlDataset

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.heads.room_furniture_vector_head import (
    RoomFurnitureVectorHead,
)
from floorflow.inference.cad_stage_runtime import CadStageRuntime
from floorflow.model.predicted_channel_checkpoint import (
    build_predicted_channel_model,
)

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vector-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--layer-checkpoint",
        type=Path,
        default=ROOT / "outputs/train/cad_furniture_dense_fast64_v1/best.pt",
    )
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    cli = parse_args()
    device = torch.device(cli.device)
    layer_payload = torch.load(
        cli.layer_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    vector_payload = torch.load(
        cli.vector_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    layer_config = layer_payload["config"]
    vector_config = vector_payload["config"]
    taxonomy = FloorplanTaxonomy.from_json(layer_payload["taxonomy"])
    layer = build_predicted_channel_model(taxonomy, layer_config).to(device)
    layer.load_state_dict(layer_payload["model"])
    layer.requires_grad_(False).eval()
    stage_runtime = CadStageRuntime(
        taxonomy,
        device,
        room_checkpoint=layer_config["room_checkpoint"],
        structure_checkpoint=layer_config["structure_checkpoint"],
        centerline_checkpoint=layer_config["centerline_checkpoint"],
        opening_checkpoint=layer_config["opening_checkpoint"],
        room_blend=float(layer_config["room_blend"]),
        structure_size=128,
    )
    head = RoomFurnitureVectorHead(
        int(layer_config["field_channels"]),
        len(taxonomy.room_ids),
        len(taxonomy.furniture_ids),
        hidden_dim=int(vector_config["hidden_dim"]),
        max_per_room=int(vector_config["max_per_room"]),
    ).to(device)
    head.load_state_dict(vector_payload["head"])
    head.eval()
    dataset = RasterPredictedControlDataset(
        Path(vector_config["predicted_root"]),
        "val",
        Path(vector_config["cache_root"]),
        Path(vector_config["data_root"]),
        image_size=int(layer_config["image_size"]),
        limit=cli.limit,
    )
    loader = DataLoader(
        dataset,
        batch_size=cli.batch_size,
        shuffle=False,
        num_workers=cli.workers,
        pin_memory=device.type == "cuda",
    )

    class EvalArgs:
        predicted_root = Path(vector_config["predicted_root"])
        minimum_room_area = int(vector_config["minimum_room_area"])
        presence_threshold = float(vector_config["presence_threshold"])

    summary, cases = evaluate(
        head,
        layer,
        stage_runtime,
        loader,
        taxonomy,
        EvalArgs(),
        device,
    )
    report = {
        "schema_version": "cad-furniture-vector-eval-v1",
        "test_opened": False,
        "checkpoint": str(cli.vector_checkpoint),
        "summary": summary,
        "cases": cases,
    }
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    cli.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
