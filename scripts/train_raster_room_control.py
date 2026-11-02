#!/usr/bin/env python3
"""Train a raster-conditioned room predictor from non-room image evidence."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from floorflow.data.class_balance import load_positive_weights
from floorflow.data.dataset import (
    CubiCasaFloorplanDataset,
    CubiCasaRasterControlDataset,
)
from floorflow.data.raster_control import (
    image_normalization,
    raster_control_image,
)
from floorflow.evaluation.room_metric_suite import room_metric_suite
from floorflow.heads.predicted_room_identity import room_identity_loss
from floorflow.heads.raster_control_bank import build_raster_room_model

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--initial-checkpoint",
        type=Path,
        default=None,
    )
    parser.add_argument("--data-root", type=Path, default=ROOT / ".data/cubicasa5k")
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=ROOT / "outputs/cache/cubicasa5k_full_v2",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/train/raster_room_control_v1",
    )
    parser.add_argument("--limit-train", type=int, default=1024)
    parser.add_argument("--limit-val", type=int, default=191)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--decoder-channels", type=int, default=96)
    parser.add_argument(
        "--architecture",
        choices=("resnet18_fpn", "deeplabv3_resnet50", "convnext_tiny_fpn"),
        default="resnet18_fpn",
    )
    parser.add_argument(
        "--backbone-weights",
        type=Path,
        default=Path(
            "/home/hugo/.cache/torch/hub/checkpoints/"
            "resnet50-0676ba61.pth"
        ),
    )
    parser.add_argument(
        "--input-rendering",
        choices=("native", "line", "nonroom"),
        default="native",
    )
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--dice-weight", type=float, default=0.5)
    parser.add_argument("--boundary-weight", type=float, default=0.5)
    parser.add_argument("--class-balance", action="store_true")
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2903)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def raster_batch(
    target: torch.Tensor,
    footprint: torch.Tensor,
    taxonomy,
) -> torch.Tensor:
    images = [
        image_normalization(
            raster_control_image(
                row.detach().cpu().numpy(),
                mask.detach().cpu().numpy(),
                taxonomy,
            )
        )
        for row, mask in zip(target, footprint, strict=True)
    ]
    return torch.from_numpy(np.stack(images)).to(target.device)


def selection_score(metrics: dict[str, float]) -> float:
    return (
        float(metrics["room_macro_iou"])
        + 0.25 * float(metrics["instance_room_f1"])
        + 0.1 * float(metrics["instance_type_accuracy"])
    )


@torch.inference_mode()
def evaluate(
    model,
    loader,
    taxonomy,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    room_ids = torch.tensor(taxonomy.room_ids, device=device)
    rows = []
    for batch in loader:
        if len(batch) == 5:
            target, footprint, control, _floor_ids, _names = batch
            control = control.to(device)
        else:
            target, footprint, _floor_ids, _names = batch
            target = target.to(device)
            footprint = footprint.to(device)
            control = raster_batch(target, footprint, taxonomy)
        target = target.to(device)
        footprint = footprint.to(device)
        logits = model(footprint, control_image=control)
        labels = room_ids[logits.argmax(dim=1)] * footprint.long()
        for index in range(target.shape[0]):
            support = torch.isin(labels[index], room_ids)
            target_support = torch.isin(target[index], room_ids)
            suite = room_metric_suite(
                labels[index].cpu().numpy(),
                target[index].cpu().numpy(),
                footprint[index].cpu().numpy(),
                taxonomy,
                predicted_room_support=support.cpu().numpy(),
                predicted_instance_support=support.cpu().numpy(),
                target_room_support=target_support.cpu().numpy(),
            )
            instance = suite["matched_instance"]
            rows.append(
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
                    "room_support_iou": suite[
                        "occlusion_aware_support"
                    ]["room_support_iou"],
                }
            )
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_class = (
        CubiCasaRasterControlDataset
        if args.input_rendering in {"native", "line"}
        else CubiCasaFloorplanDataset
    )
    dataset_kwargs = (
        {"image_mode": args.input_rendering}
        if dataset_class is CubiCasaRasterControlDataset
        else {}
    )
    train_dataset = dataset_class(
        args.data_root,
        "train",
        image_size=args.image_size,
        cache_root=args.cache_root,
        augment=True,
        limit=args.limit_train,
        **dataset_kwargs,
    )
    taxonomy = train_dataset.taxonomy
    val_dataset = dataset_class(
        args.data_root,
        "val",
        image_size=args.image_size,
        cache_root=args.cache_root,
        augment=False,
        limit=args.limit_val,
        **dataset_kwargs,
    )
    initial_checkpoint = args.initial_checkpoint
    if initial_checkpoint is None and args.architecture == "resnet18_fpn":
        initial_checkpoint = (
            ROOT
            / "outputs/train/room_segmenter_r18_uniform_full40_v2/best.pt"
        )
    payload = (
        torch.load(
            initial_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        if initial_checkpoint is not None
        else None
    )
    decoder_channels = int(
        (payload or {}).get("config", {}).get(
            "decoder_channels",
            args.decoder_channels,
        )
    )
    model = build_raster_room_model(
        args.architecture,
        len(taxonomy.room_ids),
        decoder_channels=decoder_channels,
        backbone_weights=(
            None
            if args.architecture in {"convnext_tiny_fpn", "deeplabv3_resnet50"}
            else args.backbone_weights
        ),
    ).to(device)
    if payload is not None:
        model.load_state_dict(payload["model"])
    class_weights = None
    if args.class_balance:
        positive = torch.from_numpy(
            load_positive_weights(
                args.cache_root,
                taxonomy.num_target_channels,
                20.0,
            )
        ).to(device)
        selected = positive[
            torch.tensor(
                [class_id - 1 for class_id in taxonomy.room_ids],
                device=device,
            )
        ]
        class_weights = selected / selected.mean().clamp_min(1.0e-6)
    optimizer = torch.optim.AdamW(
        model.parameters(),
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
        model.train()
        running = 0.0
        seen = 0
        for batch in train_loader:
            if len(batch) == 5:
                target, footprint, control, _floor_ids, _names = batch
                control = control.to(device)
            else:
                target, footprint, _floor_ids, _names = batch
                target = target.to(device)
                footprint = footprint.to(device)
                control = raster_batch(target, footprint, taxonomy)
            target = target.to(device)
            footprint = footprint.to(device)
            logits = model(footprint, control_image=control)
            loss, components = room_identity_loss(
                logits,
                target,
                footprint,
                taxonomy.room_ids,
                class_weights=class_weights,
                dice_weight=args.dice_weight,
                boundary_weight=args.boundary_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                args.gradient_clip,
            )
            optimizer.step()
            running += float(loss.detach()) * target.shape[0]
            seen += target.shape[0]
        validation = evaluate(model, val_loader, taxonomy, device)
        row = {
            "epoch": epoch,
            "train_loss": running / max(1, seen),
            **{key: float(value) for key, value in components.items()},
            **validation,
            "selection_score": selection_score(validation),
        }
        history.append(row)
        with (args.output_dir / "train.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        state = {
            "model": model.state_dict(),
            "config": {
                **vars(args),
                "decoder_channels": decoder_channels,
                "architecture": args.architecture,
                "input_protocol": (
                    f"{args.input_rendering}_cubicasa_raster"
                    if args.input_rendering != "nonroom"
                    else "non_room_annotation_rendering"
                ),
                "room_labels_in_input": False,
            },
            "taxonomy": taxonomy.to_json(),
            "metrics": validation,
        }
        torch.save(state, args.output_dir / "last.pt")
        if row["selection_score"] > best_score:
            best_score = row["selection_score"]
            torch.save(state, args.output_dir / "best.pt")
        print(json.dumps(row), flush=True)
    report = {
        "status": "complete",
        "scope": "CubiCasa raster-conditioned reconstruction screen",
                "input_protocol": (
                    f"{args.input_rendering}_cubicasa_raster"
                    if args.input_rendering != "nonroom"
                    else "non_room_annotation_rendering"
                ),
        "room_labels_in_input": False,
        "train_cases": len(train_dataset),
        "validation_cases": len(val_dataset),
        "best_selection_score": best_score,
        "history": history,
        "config": {
            **vars(args),
            "decoder_channels": decoder_channels,
            "architecture": args.architecture,
            "input_protocol": (
                f"{args.input_rendering}_cubicasa_raster"
                if args.input_rendering != "nonroom"
                else "non_room_annotation_rendering"
            ),
            "room_labels_in_input": False,
        },
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, default=str))


__all__ = ["evaluate", "main", "raster_batch", "selection_score"]


if __name__ == "__main__":
    main()
