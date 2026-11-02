#!/usr/bin/env python3
"""Train room-slot furniture vector primitives on frozen FloorFlow controls."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_raster_predicted_control_bridge import RasterPredictedControlDataset
from train_residential_predicted_refiner import _predicted_controls

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.furniture_vector_metrics import furniture_vector_metrics
from floorflow.geometry.wall_graph_readout import ReadoutFurniture
from floorflow.heads.dynamic_object_instance import object_instance_targets
from floorflow.heads.room_furniture_vector_head import (
    RoomFurnitureVectorHead,
)
from floorflow.inference.cad_stage_runtime import CadStageRuntime
from floorflow.inference.cad_wall_candidate_runtime import canonical_room_rows
from floorflow.inference.furniture_vector_readout import clip_furniture_to_rooms
from floorflow.model.predicted_channel_checkpoint import (
    build_predicted_channel_model,
)
from floorflow.training.cad_furniture_vector_training import (
    furniture_vector_target_loss,
    furniture_vector_targets,
)

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--predicted-root",
        type=Path,
        default=ROOT / "outputs/control_contracts/model_predicted_full_v1",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / ".data/cubicasa5k",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=ROOT / "outputs/cache/cubicasa5k_full_v2_384",
    )
    parser.add_argument("--limit-train", type=int, default=64)
    parser.add_argument("--limit-val", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--max-per-room", type=int, default=6)
    parser.add_argument("--render-grid", type=int, default=64)
    parser.add_argument("--render-weight", type=float, default=0.5)
    parser.add_argument(
        "--class-balance",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--presence-threshold", type=float, default=0.2)
    parser.add_argument("--minimum-room-area", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--seed", type=int, default=2905)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dense-control",
        type=Path,
        default=ROOT / "outputs/reports/cad_furniture_dense_fast64_eval_fthr01_v1.json",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _predicted_furniture(
    predictions: dict[str, torch.Tensor],
    room_rows: list[dict],
    taxonomy: FloorplanTaxonomy,
    *,
    frame: int,
    presence_threshold: float,
) -> tuple[ReadoutFurniture, ...]:
    exists = predictions["furniture_exists_logits"].sigmoid()
    classes = predictions["furniture_class_logits"].argmax(dim=-1)
    polygons = predictions["furniture_polygons"]
    room_labels = {
        row["label"]: index
        for index, row in enumerate(room_rows)
    }
    furniture = []
    for room_index, room_id in enumerate(taxonomy.room_ids):
        target_room = room_labels.get(taxonomy.class_names[room_id])
        if target_room is None:
            continue
        for slot in range(exists.shape[1]):
            if float(exists[room_index, slot]) < float(presence_threshold):
                continue
            polygon = polygons[room_index, slot].detach().cpu().numpy()
            polygon = polygon * float(frame - 1)
            furniture.append(
                ReadoutFurniture(
                    label=taxonomy.class_names[
                        taxonomy.furniture_ids[int(classes[room_index, slot])]
                    ],
                    polygon=tuple(
                        (float(point[0]), float(point[1]))
                        for point in polygon
                    ),
                    room_index=target_room,
                    confidence=float(exists[room_index, slot]),
                )
            )
    return clip_furniture_to_rooms(tuple(furniture), room_rows)


def _aggregate(
    rows: list[dict[str, float | str]],
) -> dict[str, float]:
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]
        if isinstance(rows[0][key], (int, float))
    }


def _class_weights(dataset, taxonomy: FloorplanTaxonomy, device: torch.device):
    counts = torch.zeros(
        len(taxonomy.furniture_ids),
        dtype=torch.float64,
    )
    for index in range(len(dataset)):
        target, _footprint, _floor_ids, _name, _image = dataset[index]
        classes, _masks, valid = object_instance_targets(
            target[None],
            taxonomy.furniture_ids,
            max_instances=128,
        )
        counts += torch.bincount(
            classes[valid].cpu(),
            minlength=len(taxonomy.furniture_ids),
        ).double()
    weights = (
        counts.sum()
        / (counts.clamp_min(1.0) * max(1, len(counts)))
    ).sqrt().clamp(0.25, 10.0)
    return (weights / weights.mean()).float().to(device), counts


@torch.inference_mode()
def evaluate(
    head,
    layer,
    stage_runtime,
    loader,
    taxonomy,
    args,
    device,
) -> tuple[dict[str, float], list[dict[str, float | str]]]:
    head.eval()
    rows = []
    for target, footprint, floor_ids, names, image in loader:
        target = target.to(device)
        footprint = footprint.to(device)
        controls = _predicted_controls(
            args.predicted_root,
            "val",
            names,
            args,
            device,
        )
        controls, _ = stage_runtime(
            controls,
            footprint,
            image.to(device),
            floor_ids.to(device),
        )
        output = layer(controls, footprint)
        predictions = head(
            output["field_features"].detach(),
            output["stage_rooms_logits"].detach(),
            render_grid=None,
        )
        resized_layers = output["cad_layers"].resized(tuple(target.shape[-2:]))
        for index in range(target.shape[0]):
            _labels, _support, rooms = canonical_room_rows(
                resized_layers.room_logits[index],
                footprint[index],
                taxonomy,
                minimum_area=args.minimum_room_area,
                largest_component_only=True,
            )
            furniture = _predicted_furniture(
                {
                    "furniture_exists_logits": predictions[
                        "furniture_exists_logits"
                    ][index],
                    "furniture_class_logits": predictions[
                        "furniture_class_logits"
                    ][index],
                    "furniture_polygons": predictions[
                        "furniture_polygons"
                    ][index],
                },
                rooms,
                taxonomy,
                frame=int(target.shape[-1]),
                presence_threshold=args.presence_threshold,
            )
            metrics = furniture_vector_metrics(
                furniture,
                rooms,
                target[index].detach().cpu().numpy(),
                taxonomy,
            )
            metrics["predicted_room_slots"] = float(
                (predictions["furniture_exists_logits"][index].sigmoid() >= args.presence_threshold)
                .sum()
            )
            metrics["case"] = str(names[index])
            rows.append(metrics)
    return _aggregate(rows), rows


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    payload = torch.load(
        args.layer_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    config = payload["config"]
    taxonomy = FloorplanTaxonomy.from_json(payload["taxonomy"])
    layer = build_predicted_channel_model(taxonomy, config).to(device)
    layer.load_state_dict(payload["model"])
    layer.requires_grad_(False).eval()
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
    common = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(train, shuffle=True, **common)
    val_loader = DataLoader(val, shuffle=False, **common)
    head = RoomFurnitureVectorHead(
        int(config["field_channels"]),
        len(taxonomy.room_ids),
        len(taxonomy.furniture_ids),
        hidden_dim=args.hidden_dim,
        max_per_room=args.max_per_room,
    ).to(device)
    class_weights, class_counts = (
        _class_weights(train, taxonomy, device)
        if args.class_balance
        else (None, torch.zeros(0))
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_vector = float("-inf")
    for epoch in range(args.epochs):
        started = time.perf_counter()
        head.train()
        totals = {}
        seen = 0
        for target, footprint, floor_ids, names, image in train_loader:
            target = target.to(device)
            footprint = footprint.to(device)
            controls = _predicted_controls(
                args.predicted_root,
                "train",
                names,
                args,
                device,
            )
            controls, _ = stage_runtime(
                controls,
                footprint,
                image.to(device),
                floor_ids.to(device),
            )
            with torch.no_grad():
                output = layer(controls, footprint)
                features = output["field_features"].detach()
                room_logits = output["stage_rooms_logits"].detach()
            target_rows = furniture_vector_targets(
                target,
                taxonomy,
                max_per_room=args.max_per_room,
            )
            predictions = head(
                features,
                room_logits,
                render_grid=args.render_grid,
            )
            loss, components = furniture_vector_target_loss(
                predictions,
                target_rows,
                render_grid=args.render_grid,
                render_weight=args.render_weight,
                class_weights=class_weights,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            batch = target.shape[0]
            seen += batch
            totals["loss"] = totals.get("loss", 0.0) + float(loss.detach()) * batch
            for key, value in components.items():
                totals[key] = totals.get(key, 0.0) + float(value) * batch
        metrics, _case_metrics = evaluate(
            head,
            layer,
            stage_runtime,
            val_loader,
            taxonomy,
            args,
            device,
        )
        record = {
            "epoch": epoch,
            **{key: value / max(1, seen) for key, value in totals.items()},
            **{f"val_{key}": value for key, value in metrics.items()},
            "seconds": time.perf_counter() - started,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        checkpoint = {
            "head": head.state_dict(),
            "config": vars(args),
            "taxonomy": taxonomy.to_json(),
            "epoch": epoch,
            "metrics": metrics,
        }
        torch.save(checkpoint, args.output_dir / "last.pt")
        if metrics["furniture_vector_f1"] > best_vector:
            best_vector = metrics["furniture_vector_f1"]
            torch.save(checkpoint, args.output_dir / "best.pt")
    best = torch.load(
        args.output_dir / "best.pt",
        map_location=device,
        weights_only=False,
    )
    head.load_state_dict(best["head"])
    evaluation, case_metrics = evaluate(
        head,
        layer,
        stage_runtime,
        val_loader,
        taxonomy,
        args,
        device,
    )
    dense = json.loads(args.dense_control.read_text(encoding="utf-8"))["summary"]
    report = {
        "schema_version": "cad-furniture-vector-v1",
        "task": "PC-06P",
        "test_opened": False,
        "source_episode": str(args.layer_checkpoint),
        "selected_epoch": int(best["epoch"]),
        "history": history,
        "evaluation": evaluation,
        "case_metrics": case_metrics,
        "dense_control": {
            "furniture_vector_f1": float(dense["furniture_vector_f1"]),
            "furniture_vector_instances": float(
                dense.get(
                    "furniture_vector_instances",
                    dense.get("furniture_vector_count", 0.0),
                )
            ),
        },
        "vector_f1_delta": (
            evaluation["furniture_vector_f1"]
            - float(dense["furniture_vector_f1"])
        ),
        "class_balance": {
            "enabled": bool(args.class_balance),
            "weights": class_weights.detach().cpu().tolist()
            if class_weights is not None
            else [],
            "counts": class_counts.tolist(),
        },
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["evaluation"], indent=2))


if __name__ == "__main__":
    main()
