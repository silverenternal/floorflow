#!/usr/bin/env python3
"""Train a vector topology scaffold on frozen CAD layer features."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scipy.optimize import linear_sum_assignment
from train_raster_predicted_control_bridge import RasterPredictedControlDataset
from train_residential_predicted_refiner import _predicted_controls
from train_vector_primitive_smoke import (
    _class_loss,
    _detection_metrics,
    _exists_loss,
    _match,
    _matched_loss,
)

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.data.vector_primitive_target_bank import (
    VectorPrimitiveTargetBank,
)
from floorflow.geometry.wall_graph_readout import raster_labels_to_wall_plan
from floorflow.heads.vector_primitive_head import VectorPrimitiveHead
from floorflow.inference.cad_stage_runtime import CadStageRuntime
from floorflow.model.predicted_channel_checkpoint import (
    build_predicted_channel_model,
)
from floorflow.objectives.vector_primitive_loss import (
    aligned_host_targets,
    host_accuracy,
    host_loss,
    matched_wall_loss,
)

ROOT = Path(__file__).resolve().parents[1]
GRAPH_STAGES = (
    "envelope",
    "program",
    "rooms",
    "walls",
    "junctions",
    "openings",
    "furniture",
    "validation",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer-checkpoint", type=Path, required=True)
    parser.add_argument("--predicted-root", type=Path, default=ROOT / "outputs/control_contracts/model_predicted_full_v1")
    parser.add_argument("--data-root", type=Path, default=ROOT / ".data/cubicasa5k")
    parser.add_argument("--cache-root", type=Path, default=ROOT / "outputs/cache/cubicasa5k_full_v2_384")
    parser.add_argument("--geometry-root", type=Path, default=ROOT / "outputs/cache/cubicasa5k_geometry_v1")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit-train", type=int, default=256)
    parser.add_argument("--limit-val", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--room-queries", type=int, default=24)
    parser.add_argument("--max-walls", type=int, default=64)
    parser.add_argument("--max-openings", type=int, default=48)
    parser.add_argument(
        "--vector-targets",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--stage-context", action="store_true")
    parser.add_argument("--host-conditioned-openings", action="store_true")
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2905)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def target_vectors(
    labels: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
    *,
    max_walls: int,
    max_openings: int,
) -> dict[str, torch.Tensor]:
    """Convert target labels to padded wall/opening vector targets."""

    batch, height, width = labels.shape
    scale = float(max(height, width))
    wall_endpoints = torch.zeros(batch, max_walls, 2, 2)
    wall_exists = torch.zeros(batch, max_walls)
    wall_thickness = torch.zeros(batch, max_walls)
    wall_exterior = torch.zeros(batch, max_walls)
    opening_centers = torch.zeros(batch, max_openings, 2)
    opening_exists = torch.zeros(batch, max_openings)
    opening_widths = torch.zeros(batch, max_openings)
    opening_classes = torch.zeros(batch, max_openings, dtype=torch.long)
    opening_lookup = {
        class_id: index
        for index, class_id in enumerate(taxonomy.opening_ids)
    }
    door_id = next(
        class_id
        for class_id in taxonomy.opening_ids
        if taxonomy.class_names[class_id].startswith("door:")
    )
    window_id = next(
        class_id
        for class_id in taxonomy.opening_ids
        if taxonomy.class_names[class_id].startswith("window:")
    )
    for index, label_map in enumerate(labels.cpu().numpy()):
        plan = raster_labels_to_wall_plan(label_map, taxonomy)
        wall_count = min(len(plan.walls), max_walls)
        for item, wall in enumerate(plan.walls[:wall_count]):
            wall_endpoints[index, item] = torch.tensor(
                [
                    [wall.start[0] / scale, wall.start[1] / scale],
                    [wall.end[0] / scale, wall.end[1] / scale],
                ]
            )
            wall_exists[index, item] = 1.0
            wall_thickness[index, item] = float(wall.thickness) / scale
            wall_exterior[index, item] = float(wall.wall_type == "exterior")
        opening_count = min(len(plan.openings), max_openings)
        for item, opening in enumerate(plan.openings[:opening_count]):
            opening_centers[index, item] = torch.tensor(
                [opening.center[0] / scale, opening.center[1] / scale]
            )
            opening_exists[index, item] = 1.0
            opening_widths[index, item] = float(opening.width) / scale
            class_id = door_id if opening.kind == "door" else window_id
            opening_classes[index, item] = opening_lookup.get(class_id, 0)
    return {
        "wall_endpoints": wall_endpoints,
        "wall_exists": wall_exists,
        "wall_thickness": wall_thickness,
        "wall_exterior": wall_exterior,
        "opening_centers": opening_centers,
        "opening_exists": opening_exists,
        "opening_widths": opening_widths,
        "opening_classes": opening_classes,
    }


def scaffold_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, float]]:
    predicted_walls = outputs["wall_endpoints"]
    target_walls = targets["wall_endpoints"]
    wall_pairs = []
    for batch in range(predicted_walls.shape[0]):
        count = int(targets["wall_exists"][batch].sum())
        if count == 0:
            wall_pairs.append(
                (
                    np.zeros(0, dtype=np.int64),
                    np.zeros(0, dtype=np.int64),
                )
            )
            continue
        direct = (
            predicted_walls[batch][:, None]
            - target_walls[batch, :count][None]
        ).abs().mean(dim=(2, 3))
        reverse = (
            predicted_walls[batch][:, None]
            - target_walls[batch, :count][None, :, [1, 0]]
        ).abs().mean(dim=(2, 3))
        predicted_length = (
            predicted_walls[batch][:, 1] - predicted_walls[batch][:, 0]
        ).norm(dim=-1)
        target_length = (
            target_walls[batch, :count, 1] - target_walls[batch, :count, 0]
        ).norm(dim=-1)
        length_cost = (
            predicted_length[:, None] - target_length[None]
        ).abs()
        cost = torch.minimum(direct, reverse) + 0.5 * length_cost
        rows, columns = linear_sum_assignment(cost.detach().cpu().numpy())
        wall_pairs.append((rows.astype(np.int64), columns.astype(np.int64)))
    opening_pairs = _match(
        outputs["opening_centers"],
        targets["opening_centers"],
        targets["opening_exists"].bool(),
        class_logits=outputs["opening_class"],
        target_classes=targets["opening_classes"],
    )
    aligned_opening_hosts = aligned_host_targets(
        outputs.get("opening_host_logits"),
        targets.get("opening_hosts"),
        wall_pairs,
        opening_pairs,
    )
    wall_loss = (
        _exists_loss(outputs["wall_exists"], wall_pairs)
        + 5.0 * matched_wall_loss(
            outputs["wall_endpoints"],
            targets["wall_endpoints"],
            wall_pairs,
        )
        + torch.nn.functional.smooth_l1_loss(
            outputs["wall_thickness"].sigmoid(),
            targets["wall_thickness"],
        )
        + torch.nn.functional.binary_cross_entropy_with_logits(
            outputs["wall_exterior"],
            targets["wall_exterior"],
        )
    )
    opening_loss = (
        _exists_loss(outputs["opening_exists"], opening_pairs)
        + 5.0
        * _matched_loss(
            outputs["opening_centers"],
            targets["opening_centers"],
            opening_pairs,
        )
        + torch.nn.functional.smooth_l1_loss(
            outputs["opening_widths"].sigmoid(),
            targets["opening_widths"],
        )
        + 0.5
        * _class_loss(
            outputs["opening_class"],
            targets["opening_classes"],
            opening_pairs,
        )
        + host_loss(
            outputs.get("opening_host_logits"),
            aligned_opening_hosts,
            opening_pairs,
            outputs["opening_exists"],
        )
    )
    metrics = {
        **{
            f"wall_{key}": value
            for key, value in _detection_metrics(
                outputs["wall_exists"],
                wall_pairs,
                outputs["wall_endpoints"],
                targets["wall_endpoints"],
            ).items()
        },
        **{
            f"opening_{key}": value
            for key, value in _detection_metrics(
                outputs["opening_exists"],
                opening_pairs,
                outputs["opening_centers"],
                targets["opening_centers"],
            ).items()
        },
        "opening_host_accuracy": host_accuracy(
            outputs.get("opening_host_logits"),
            aligned_opening_hosts,
            opening_pairs,
        ),
    }
    return wall_loss + opening_loss, {
        "vector_wall_loss": wall_loss.detach(),
        "vector_opening_loss": opening_loss.detach(),
    }, metrics


def run_epoch(
    scaffold,
    layer_model,
    stage_runtime,
    graph_projection,
    loader,
    taxonomy,
    args,
    device,
    optimizer=None,
    target_bank: VectorPrimitiveTargetBank | None = None,
) -> dict[str, float]:
    scaffold.train(optimizer is not None)
    totals = {}
    seen = 0
    for target, footprint, floor_ids, names, image in loader:
        target = target.to(device)
        footprint = footprint.to(device)
        controls = _predicted_controls(
            args.predicted_root,
            "train" if optimizer is not None else "val",
            names,
            args,
            device,
        )
        controls, _diagnostics = stage_runtime(
            controls,
            footprint,
            image.to(device),
            floor_ids.to(device),
        )
        with torch.no_grad():
            layer_output = layer_model(controls, footprint)
            features = layer_output["field_features"]
            tokens = torch.stack(
                [layer_output["action_tokens"][name] for name in GRAPH_STAGES],
                dim=1,
            )
            graph_mask = torch.ones(
                tokens.shape[:2],
                dtype=torch.bool,
                device=device,
            )
        outputs = scaffold(
            features,
            graph_tokens=graph_projection(tokens),
            graph_mask=graph_mask,
        )
        targets = (
            target_bank.batch_targets(names, device)
            if target_bank is not None
            else {
                key: value.to(device)
                for key, value in target_vectors(
                    target,
                    taxonomy,
                    max_walls=args.max_walls,
                    max_openings=args.max_openings,
                ).items()
            }
        )
        loss, components, metrics = scaffold_loss(outputs, targets)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [*scaffold.parameters(), *graph_projection.parameters()],
                1.0,
            )
            optimizer.step()
        totals["loss"] = totals.get("loss", 0.0) + float(loss.detach()) * target.shape[0]
        for key, value in components.items():
            totals[key] = totals.get(key, 0.0) + float(value) * target.shape[0]
        for key, value in metrics.items():
            scale = 1.0 if key.rsplit("_", 1)[-1] in {
                "true_positive",
                "positive",
                "predicted",
                "targets",
            } else target.shape[0]
            totals[key] = totals.get(key, 0.0) + float(value) * scale
        seen += target.shape[0]
    return {key: value / max(1, seen) for key, value in totals.items()}


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    payload = torch.load(args.layer_checkpoint, map_location="cpu", weights_only=False)
    config = payload["config"]
    taxonomy = FloorplanTaxonomy.from_json(payload["taxonomy"])
    layer_model = build_predicted_channel_model(taxonomy, config).to(device)
    layer_model.load_state_dict(payload["model"])
    layer_model.requires_grad_(False).eval()
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
    scaffold = VectorPrimitiveHead(
        int(config["field_channels"]),
        len(taxonomy.room_ids),
        len(taxonomy.opening_ids),
        hidden_dim=args.hidden_dim,
        max_rooms=args.room_queries,
        max_walls=args.max_walls,
        max_openings=args.max_openings,
        polygon_points=8,
        decoder_layers=2,
        attention_heads=4,
        stage_context=args.stage_context,
        host_conditioned_openings=args.host_conditioned_openings,
    ).to(device)
    graph_projection = torch.nn.Linear(
        int(config["field_channels"]),
        args.hidden_dim,
    ).to(device)
    train = RasterPredictedControlDataset(
        args.predicted_root,
        "train",
        args.cache_root,
        args.data_root,
        image_size=int(config["image_size"]),
        limit=args.limit_train,
    )
    val = RasterPredictedControlDataset(
        args.predicted_root,
        "val",
        args.cache_root,
        args.data_root,
        image_size=int(config["image_size"]),
        limit=args.limit_val,
    )
    loader_args = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": True,
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(train, shuffle=True, **loader_args)
    val_loader = DataLoader(val, shuffle=False, **loader_args)
    target_bank = (
        VectorPrimitiveTargetBank(
            args.geometry_root,
            "train",
            image_size=int(config["image_size"]),
            max_walls=args.max_walls,
            max_openings=args.max_openings,
        )
        if args.vector_targets
        else None
    )
    val_target_bank = (
        VectorPrimitiveTargetBank(
            args.geometry_root,
            "val",
            image_size=int(config["image_size"]),
            max_walls=args.max_walls,
            max_openings=args.max_openings,
        )
        if args.vector_targets
        else None
    )
    parameters = [*scaffold.parameters(), *graph_projection.parameters()]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best = -1.0
    for epoch in range(args.epochs):
        train_metrics = run_epoch(
            scaffold,
            layer_model,
            stage_runtime,
            graph_projection,
            train_loader,
            taxonomy,
            args,
            device,
            optimizer,
            target_bank,
        )
        val_metrics = run_epoch(
            scaffold,
            layer_model,
            stage_runtime,
            graph_projection,
            val_loader,
            taxonomy,
            args,
            device,
            target_bank=val_target_bank,
        )
        score = val_metrics["wall_f1"] + val_metrics["opening_f1"]
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "selection_score": score,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        checkpoint = {
            "scaffold": scaffold.state_dict(),
            "graph_projection": graph_projection.state_dict(),
            "taxonomy": taxonomy.to_json(),
            "layer_checkpoint": str(args.layer_checkpoint),
            "config": {**vars(args), "field_channels": int(config["field_channels"])},
            "metrics": val_metrics,
        }
        torch.save(checkpoint, args.output_dir / "last.pt")
        if score > best:
            best = score
            torch.save(checkpoint, args.output_dir / "best.pt")
    (args.output_dir / "report.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "scope": "vector topology scaffold on predicted controls",
                "test_opened": False,
                "target_used_at_inference": False,
                "history": history,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
