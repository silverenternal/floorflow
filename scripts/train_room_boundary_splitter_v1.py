#!/usr/bin/env python3
"""Train a frozen-room boundary head for instance splitting."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from floorflow.data.dataset import CubiCasaRasterControlDataset
from floorflow.evaluation.room_metric_suite import room_metric_suite
from floorflow.heads.room_instance_boundary import (
    BoundaryAwareRoomSegmenter,
    room_boundary_loss,
    room_boundary_metrics,
    room_instance_boundary_target,
    split_room_instances,
)

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / ".data/cubicasa5k")
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=ROOT / "outputs/cache/cubicasa5k_full_v2",
    )
    parser.add_argument(
        "--room-checkpoint",
        type=Path,
        default=ROOT / "outputs/train/raster_room_control_native_full_v1/best.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/train/room_boundary_splitter_screen256_v1",
    )
    parser.add_argument("--limit-train", type=int, default=256)
    parser.add_argument("--limit-val", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--decoder-channels", type=int, default=128)
    parser.add_argument("--boundary-channels", type=int, default=16)
    parser.add_argument(
        "--boundary-mode",
        choices=("final", "multiscale", "independent", "dense"),
        default="dense",
    )
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--positive-weight", type=float, default=4.0)
    parser.add_argument("--tversky-weight", type=float, default=0.0)
    parser.add_argument("--tversky-alpha", type=float, default=0.3)
    parser.add_argument("--tversky-beta", type=float, default=0.7)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2905)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def selection_score(metrics: dict[str, float]) -> float:
    return (
        metrics["room_iou"]
        + 0.75 * metrics["instance_f1"]
        + 0.1 * metrics["type_accuracy"]
        - 0.02 * metrics["count_mae"]
    )


@torch.inference_mode()
def evaluate(model, loader, taxonomy, device) -> dict:
    thresholds = (0.3, 0.5, 0.7, 0.8, 0.9)
    rows = {"baseline": []}
    rows.update({f"clear_{threshold:.1f}": [] for threshold in thresholds})
    boundary_rows = {threshold: [] for threshold in thresholds}
    room_ids = torch.tensor(taxonomy.room_ids, device=device)
    for target, footprint, image, _floor_ids, _names in loader:
        target = target.to(device)
        footprint = footprint.to(device)
        image = image.to(device)
        output = model(footprint, control_image=image)
        labels = room_ids[output["dense_logits"].argmax(dim=1)]
        labels *= footprint.long()
        predictions = {"baseline": labels}
        for threshold in thresholds:
            predictions[f"clear_{threshold:.1f}"] = split_room_instances(
                labels,
                output["boundary_logits"],
                footprint,
                room_ids,
                threshold=threshold,
            )
        target_boundary = room_instance_boundary_target(
            target,
            footprint,
            taxonomy,
        )
        for threshold in thresholds:
            boundary_rows[threshold].append(
                room_boundary_metrics(
                    output["boundary_logits"],
                    target_boundary,
                    footprint,
                    threshold=threshold,
                )
            )
        for index in range(target.shape[0]):
            predicted_support = torch.isin(
                predictions["baseline"][index],
                room_ids,
            )
            target_support = torch.isin(target[index], room_ids)
            for name, prediction in predictions.items():
                suite = room_metric_suite(
                    prediction[index].cpu().numpy(),
                    target[index].cpu().numpy(),
                    footprint[index].cpu().numpy(),
                    taxonomy,
                    predicted_room_support=predicted_support.cpu().numpy(),
                    target_room_support=target_support.cpu().numpy(),
                )
                instance = suite["matched_instance"]
                rows[name].append(
                    {
                        "room_iou": suite["visible_semantic"]["room_macro_iou"],
                        "instance_f1": instance["instance_f1_at_threshold"],
                        "type_accuracy": instance["matched_type_accuracy"],
                        "count_mae": instance["count_abs_error"],
                        "room_support_iou": suite[
                            "occlusion_aware_support"
                        ]["room_support_iou"],
                    }
                )
    arms = {
        name: {
            key: float(np.mean([row[key] for row in values]))
            for key in values[0]
        }
        for name, values in rows.items()
    }
    boundary = {
        f"{threshold:.1f}": {
            key: float(np.mean([row[key] for row in values]))
            for key in values[0]
        }
        for threshold, values in boundary_rows.items()
    }
    return {"arms": arms, "boundary": boundary}


def main() -> None:
    args = parse_args()
    if args.device != "cuda:0":
        raise ValueError("room boundary training is restricted to cuda:0")
    seed_everything(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_dataset = CubiCasaRasterControlDataset(
        args.data_root,
        "train",
        image_size=args.image_size,
        cache_root=args.cache_root,
        augment=True,
        limit=args.limit_train,
        image_mode="native",
    )
    taxonomy = train_dataset.taxonomy
    val_dataset = CubiCasaRasterControlDataset(
        args.data_root,
        "val",
        image_size=args.image_size,
        cache_root=args.cache_root,
        augment=False,
        limit=args.limit_val,
        image_mode="native",
    )
    model = BoundaryAwareRoomSegmenter(
        taxonomy,
        decoder_channels=args.decoder_channels,
        boundary_channels=args.boundary_channels,
        boundary_mode=args.boundary_mode,
    ).to(device)
    payload = torch.load(args.room_checkpoint, map_location="cpu", weights_only=False)
    model.room.load_state_dict(payload["model"])
    model.room.requires_grad_(False)
    model.room.eval()
    boundary_parameters = sum(
        parameter.numel()
        for parameter in model.boundary.parameters()
    )
    optimizer = torch.optim.AdamW(
        model.boundary.parameters(),
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
    history = []
    best_score = -1.0
    for epoch in range(args.epochs):
        model.boundary.train()
        running = 0.0
        components_sum: dict[str, float] = {}
        seen = 0
        for target, footprint, image, _floor_ids, _names in train_loader:
            target = target.to(device)
            footprint = footprint.to(device)
            image = image.to(device)
            output = model(footprint, control_image=image)
            boundary = room_instance_boundary_target(
                target,
                footprint,
                taxonomy,
            )
            loss, components = room_boundary_loss(
                output["boundary_logits"],
                boundary,
                footprint,
                positive_weight=args.positive_weight,
                tversky_weight=args.tversky_weight,
                tversky_alpha=args.tversky_alpha,
                tversky_beta=args.tversky_beta,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.boundary.parameters(),
                args.gradient_clip,
            )
            optimizer.step()
            running += float(loss.detach()) * target.shape[0]
            seen += target.shape[0]
            for key, value in components.items():
                components_sum[key] = components_sum.get(key, 0.0) + float(
                    value.detach().cpu()
                )
        validation = evaluate(model, val_loader, taxonomy, device)
        current_name, current_metrics = max(
            validation["arms"].items(),
            key=lambda item: selection_score(item[1]),
        )
        row = {
            "epoch": epoch,
            "train_loss": running / max(1, seen),
            "train_components": {
                key: value / max(1, len(train_loader))
                for key, value in components_sum.items()
            },
            "best_arm": current_name,
            "validation": validation,
            "selection_score": selection_score(current_metrics),
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        state = {
            "model": model.state_dict(),
            "config": vars(args),
            "taxonomy": taxonomy.to_json(),
            "metrics": validation,
        }
        torch.save(state, args.output_dir / "last.pt")
        if row["selection_score"] > best_score:
            best_score = row["selection_score"]
            torch.save(state, args.output_dir / "best.pt")
    report = {
        "status": "complete",
        "scope": "learned room-instance boundary splitter",
        "condition_mode": "predicted_control",
        "target_used_at_inference": False,
        "train_cases": len(train_dataset),
        "validation_cases": len(val_dataset),
        "boundary_mode": args.boundary_mode,
        "boundary_parameters": int(boundary_parameters),
        "best_selection_score": best_score,
        "history": history,
        "config": vars(args),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
