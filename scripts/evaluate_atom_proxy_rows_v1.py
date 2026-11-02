"""Evaluate one typed-channel checkpoint into per-case atom metric rows."""

from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_raster_predicted_control_bridge import RasterPredictedControlDataset
from train_residential_predicted_refiner import _predicted_controls

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.inference.cad_stage_runtime import CadStageRuntime
from floorflow.model.predicted_channel_checkpoint import (
    build_predicted_channel_model,
)
from floorflow.training.typed_channel_validation import validate_typed_channel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predicted-root", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--limit-val", type=int)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--evaluate-deterministic-wall-source",
        action="store_true",
    )
    parser.add_argument(
        "--owner-mode",
        choices=("checkpoint", "absolute", "relative"),
        default="checkpoint",
    )
    parser.add_argument(
        "--graph-residual-mode",
        choices=(
            "checkpoint",
            "free",
            "area_preserving",
            "margin",
            "area_preserving_margin",
            "class_only",
            "bypass",
        ),
        default="checkpoint",
    )
    return parser.parse_args()


def _config(args: argparse.Namespace, config: dict) -> Namespace:
    values = dict(config)
    for name in ("predicted_root", "data_root", "cache_root"):
        value = getattr(args, name)
        if value is not None:
            values[name] = value
    if args.limit_val is not None:
        values["limit_val"] = args.limit_val
    values["batch_size"] = args.batch_size
    values["workers"] = args.workers
    values["device"] = args.device
    values["evaluate_deterministic_wall_source"] = bool(
        args.evaluate_deterministic_wall_source
    )
    return Namespace(**values)


def main() -> None:
    cli = parse_args()
    device = torch.device(cli.device)
    if device.type == "cuda" and device.index not in {None, 0}:
        raise ValueError("atom proxy evaluation is restricted to cuda:0")
    checkpoint = torch.load(
        cli.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    config = _config(cli, checkpoint["config"])
    taxonomy = FloorplanTaxonomy.read(config.cache_root / "taxonomy.json")
    dataset = RasterPredictedControlDataset(
        config.predicted_root,
        "val",
        config.cache_root,
        config.data_root,
        image_size=config.image_size,
        limit=config.limit_val,
    )
    loader = DataLoader(
        dataset,
        batch_size=cli.batch_size,
        shuffle=False,
        num_workers=cli.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=cli.workers > 0,
    )
    model_config = dict(checkpoint["config"])
    if cli.owner_mode != "checkpoint":
        model_config["owner_preserving_room_readout"] = cli.owner_mode
    if cli.graph_residual_mode != "checkpoint":
        model_config["area_preserving_mask_residual"] = (
            False
            if cli.graph_residual_mode == "free"
            else cli.graph_residual_mode
        )
    model = build_predicted_channel_model(taxonomy, model_config)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device).eval()
    stage_runtime = CadStageRuntime(
        taxonomy,
        device,
        room_checkpoint=config.room_checkpoint,
        structure_checkpoint=config.structure_checkpoint,
        centerline_checkpoint=config.centerline_checkpoint,
        opening_checkpoint=config.opening_checkpoint,
        room_blend=config.room_blend,
        structure_size=128,
    )
    rows: list[dict[str, object]] = []
    metrics = validate_typed_channel(
        model,
        loader,
        taxonomy,
        stage_runtime,
        _predicted_controls,
        config,
        device,
        epoch=int(checkpoint.get("epoch", 0)),
        row_sink=rows,
    )
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    with cli.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    report = {
        "schema_version": "atom-proxy-evaluation-v1",
        "checkpoint": str(cli.checkpoint),
        "checkpoint_epoch": int(checkpoint.get("epoch", 0)),
        "cases": len(rows),
        "target_used_at_inference": False,
        "owner_mode": cli.owner_mode,
        "evaluate_deterministic_wall_source": bool(
            cli.evaluate_deterministic_wall_source
        ),
        "graph_residual_mode": cli.graph_residual_mode,
        "metrics": metrics,
        "rows": str(cli.output),
    }
    report_path = cli.output.with_suffix(".report.json")
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "rows": str(cli.output),
                "report": str(report_path),
                "cases": len(rows),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
