#!/usr/bin/env python3
"""Train anchor-conditioned vector primitives from native raster controls."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from train_vector_primitive_smoke import _objective

from floorflow.data.raster_vector_geometry_dataset import (
    RasterVectorGeometryDataset,
)
from floorflow.evaluation.room_corner_metrics import room_corner_angle_metrics
from floorflow.evaluation.room_metric_suite import room_metric_suite
from floorflow.evaluation.structural_legality import structural_legality_metrics
from floorflow.geometry.core import compose_known_exterior
from floorflow.heads.raster_control_bank import load_raster_room_model
from floorflow.heads.structured_objects import partition_wall_mask
from floorflow.heads.vector_primitive_head import VectorPrimitiveHead
from floorflow.representation.polygon_targets import decode_polygon_loops

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--room-checkpoint",
        type=Path,
        default=ROOT / "outputs/train/raster_room_control_native_full_v1/best.pt",
    )
    parser.add_argument(
        "--geometry-root",
        type=Path,
        default=ROOT / "outputs/cache/cubicasa5k_geometry_v1",
    )
    parser.add_argument("--data-root", type=Path, default=ROOT / ".data/cubicasa5k")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/train/raster_vector_primitive_screen256_v1",
    )
    parser.add_argument("--limit-train", type=int, default=256)
    parser.add_argument("--limit-val", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--room-queries", type=int, default=24)
    parser.add_argument("--room-points", type=int, default=16)
    parser.add_argument("--decoder-layers", type=int, default=2)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2907)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def raster_features(
    room_model,
    footprint: torch.Tensor,
    image: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        hidden = room_model.encode_control(
            footprint,
            control_image=image,
        )
        logits = room_model.final(hidden)
    return torch.cat([hidden, logits.softmax(dim=1)], dim=1), logits


def structural_labels(
    room_labels: torch.Tensor,
    footprint: torch.Tensor,
    taxonomy,
) -> torch.Tensor:
    seam = partition_wall_mask(room_labels, radius=1)
    labels = torch.where(
        seam,
        torch.full_like(room_labels, taxonomy.class_id("wall")),
        room_labels,
    )
    return compose_known_exterior(
        labels,
        footprint,
        opening_ids=taxonomy.opening_ids,
    )


@torch.inference_mode()
def evaluate(
    head,
    room_model,
    loader,
    taxonomy,
    device,
    thresholds: tuple[float, ...],
) -> dict[float, dict[str, float]]:
    head.eval()
    rows = {threshold: [] for threshold in thresholds}
    for batch in loader:
        moved = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        features, _logits = raster_features(
            room_model,
            moved["footprint"],
            moved["image"],
        )
        outputs = head(features)
        for threshold in thresholds:
            room_labels = decode_polygon_loops(
                {
                    "class_logits": outputs["room_class"],
                    "polygons": outputs["room_polygons"],
                    "valid_logits": outputs["room_exists"],
                },
                taxonomy,
                height=moved["target"].shape[-2],
                width=moved["target"].shape[-1],
                validity_threshold=threshold,
            )
            labels = structural_labels(
                room_labels * moved["footprint"].long(),
                moved["footprint"],
                taxonomy,
            )
            room_ids = torch.tensor(taxonomy.room_ids, device=device)
            for index in range(labels.shape[0]):
                support = torch.isin(labels[index], room_ids)
                target_support = torch.isin(
                    moved["target"][index],
                    room_ids,
                )
                suite = room_metric_suite(
                    labels[index].cpu().numpy(),
                    moved["target"][index].cpu().numpy(),
                    moved["footprint"][index].cpu().numpy(),
                    taxonomy,
                    predicted_room_support=support.cpu().numpy(),
                    predicted_instance_support=support.cpu().numpy(),
                    target_room_support=target_support.cpu().numpy(),
                )
                instance = suite["matched_instance"]
                corner = room_corner_angle_metrics(
                    labels[index].cpu().numpy(),
                    moved["target"][index].cpu().numpy(),
                    taxonomy,
                )
                structural = structural_legality_metrics(
                    labels[index].cpu().numpy(),
                    moved["target"][index].cpu().numpy(),
                    taxonomy,
                )
                rows[threshold].append(
                    {
                        "room_macro_iou": suite["visible_semantic"][
                            "room_macro_iou"
                        ],
                        "instance_geometric_iou": instance[
                            "matched_geometric_iou"
                        ],
                        "instance_room_f1": instance[
                            "instance_f1_at_threshold"
                        ],
                        "instance_type_accuracy": instance[
                            "matched_type_accuracy"
                        ],
                        "room_closure_rate": structural[
                            "room_closure_rate"
                        ],
                        "wall_continuity_rate": structural[
                            "wall_continuity_rate"
                        ],
                        "self_intersection_rate": structural[
                            "self_intersection_rate"
                        ],
                        "node_f1": structural["node_f1"],
                        "edge_overlap": structural["edge_overlap"],
                        "ged": structural["ged"],
                        **corner,
                    }
                )
    return {
        threshold: {
            key: float(np.mean([row[key] for row in values]))
            for key in values[0]
        }
        for threshold, values in rows.items()
    }


def selection_score(metrics: dict[str, float]) -> float:
    return (
        metrics["corner_f1"]
        + metrics["angle_f1"]
        + metrics["instance_room_f1"]
        + 0.25 * metrics["room_macro_iou"]
        + 0.2 * metrics["wall_continuity_rate"]
    )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    train_dataset = RasterVectorGeometryDataset(
        args.geometry_root,
        args.data_root,
        "train",
        limit=args.limit_train,
    )
    taxonomy = train_dataset.taxonomy
    val_dataset = RasterVectorGeometryDataset(
        args.geometry_root,
        args.data_root,
        "val",
        limit=args.limit_val,
    )
    room_model = load_raster_room_model(
        args.room_checkpoint,
        taxonomy,
        device,
    )
    room_model.requires_grad_(False)
    head = VectorPrimitiveHead(
        in_channels=128 + len(taxonomy.room_ids),
        room_classes=len(taxonomy.room_ids),
        opening_classes=max(1, len(taxonomy.opening_ids)),
        hidden_dim=args.hidden_dim,
        max_rooms=args.room_queries,
        polygon_points=args.room_points,
        decoder_layers=args.decoder_layers,
        attention_heads=args.attention_heads,
    ).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    loader_args = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": True,
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_args)
    thresholds = (0.1, 0.3, 0.5)
    history = []
    best_score = -float("inf")
    best_threshold = thresholds[0]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        head.train()
        totals = {}
        seen = 0
        for batch in train_loader:
            moved = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            features, _logits = raster_features(
                room_model,
                moved["footprint"],
                moved["image"],
            )
            outputs = head(features)
            loss, components, metrics = _objective(outputs, moved)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                head.parameters(),
                args.gradient_clip,
            )
            optimizer.step()
            seen += moved["target"].shape[0]
            for key, value in {
                **components,
                **{f"primitive_{key}": value for key, value in metrics.items()},
            }.items():
                totals[key] = totals.get(key, 0.0) + float(value) * moved[
                    "target"
                ].shape[0]
        validation = evaluate(
            head,
            room_model,
            val_loader,
            taxonomy,
            device,
            thresholds,
        )
        selected_threshold, selected = max(
            validation.items(),
            key=lambda item: selection_score(item[1]),
        )
        score = selection_score(selected)
        row = {
            "epoch": epoch,
            **{f"train_{key}": value / max(1, seen) for key, value in totals.items()},
            "selected_threshold": selected_threshold,
            "selection_score": score,
            "selected": selected,
            "thresholds": {str(key): value for key, value in validation.items()},
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        state = {
            "model": head.state_dict(),
            "taxonomy": taxonomy.to_json(),
            "config": vars(args),
            "threshold": selected_threshold,
            "metrics": selected,
        }
        torch.save(state, args.output_dir / "last.pt")
        if score > best_score:
            best_score = score
            best_threshold = selected_threshold
            torch.save(state, args.output_dir / "best.pt")
    report = {
        "schema_version": "raster-vector-primitive-screen-v1",
        "scope": "mechanism screening only",
        "train_cases": len(train_dataset),
        "validation_cases": len(val_dataset),
        "test_opened": False,
        "target_used_at_inference": False,
        "best_threshold": best_threshold,
        "best_score": best_score,
        "history": history,
        "config": vars(args),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"report": str(args.output_dir / "report.json"), "best_score": best_score}, indent=2))


if __name__ == "__main__":
    main()
