#!/usr/bin/env python3
"""Train a full-semantic raster-to-CAD control bank."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from train_raster_room_control import raster_batch

from floorflow.data.class_balance import load_positive_weights
from floorflow.data.dataset import (
    CubiCasaFloorplanDataset,
    CubiCasaRasterControlDataset,
)
from floorflow.evaluation.room_metric_suite import room_metric_suite
from floorflow.heads.predicted_room_identity import room_identity_loss
from floorflow.heads.raster_control_bank import RasterControlBank
from floorflow.heads.structured_objects import (
    object_group_metrics,
    object_group_predictions,
    structured_object_loss,
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
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/train/raster_full_cad_bank_v1",
    )
    parser.add_argument(
        "--architecture",
        choices=("resnet18_fpn", "deeplabv3_resnet50"),
        default="resnet18_fpn",
    )
    parser.add_argument(
        "--room-checkpoint",
        type=Path,
        default=ROOT
        / "outputs/train/raster_room_control_native_full_v1/best.pt",
    )
    parser.add_argument("--freeze-room", action="store_true")
    parser.add_argument(
        "--input-rendering",
        choices=("native", "line", "nonroom"),
        default="native",
    )
    parser.add_argument(
        "--backbone-weights",
        type=Path,
        default=Path(
            "/home/hugo/.cache/torch/hub/checkpoints/"
            "resnet50-0676ba61.pth"
        ),
    )
    parser.add_argument("--limit-train", type=int, default=1024)
    parser.add_argument("--limit-val", type=int, default=191)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--decoder-channels", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--object-learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--dice-weight", type=float, default=0.5)
    parser.add_argument("--boundary-weight", type=float, default=0.2)
    parser.add_argument("--room-loss-weight", type=float, default=0.25)
    parser.add_argument("--object-loss-weight", type=float, default=1.0)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2904)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def data_batch(
    batch,
    taxonomy,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(batch) == 5:
        target, footprint, control, _floor_ids, _names = batch
        control = control.to(device)
    else:
        target, footprint, _floor_ids, _names = batch
        target = target.to(device)
        footprint = footprint.to(device)
        control = raster_batch(target, footprint, taxonomy)
    return target.to(device), footprint.to(device), control


def output_group_layers(outputs, index: int, taxonomy):
    sliced = {
        key: value[index : index + 1]
        for key, value in outputs.items()
        if key != "room_logits"
    }
    return object_group_predictions(sliced, taxonomy)


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
        target, footprint, control = data_batch(batch, taxonomy, device)
        outputs = model(footprint, control_image=control)
        labels = room_ids[outputs["room_logits"].argmax(dim=1)] * footprint.long()
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
            object_metrics = object_group_metrics(
                output_group_layers(outputs, index, taxonomy),
                target[index : index + 1],
                taxonomy,
            )
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
                    **object_metrics,
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
    model = RasterControlBank(
        taxonomy,
        decoder_channels=args.decoder_channels,
    ).to(device)
    room_payload = torch.load(
        args.room_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    room_channels = int(
        room_payload.get("config", {}).get(
            "decoder_channels",
            args.decoder_channels,
        )
    )
    if room_channels != args.decoder_channels:
        raise ValueError(
            "room checkpoint decoder_channels does not match bank config"
        )
    model.room.load_state_dict(room_payload["model"])
    if args.freeze_room:
        model.room.requires_grad_(False)
    positive = torch.from_numpy(
        load_positive_weights(
            args.cache_root,
            taxonomy.num_target_channels,
            20.0,
        )
    ).to(device)
    room_indices = torch.tensor(
        [class_id - 1 for class_id in taxonomy.room_ids],
        device=device,
    )
    room_weights = positive[room_indices]
    room_weights = room_weights / room_weights.mean().clamp_min(1.0e-6)
    object_parameters = list(model.object_heads.parameters())
    room_parameters = [
        parameter
        for parameter in model.room.parameters()
        if parameter.requires_grad
    ]
    parameter_groups = [
        {
            "params": object_parameters,
            "lr": (
                args.learning_rate
                if args.object_learning_rate is None
                else args.object_learning_rate
            ),
        }
    ]
    if room_parameters:
        parameter_groups.append(
            {"params": room_parameters, "lr": args.learning_rate}
        )
    optimizer = torch.optim.AdamW(
        parameter_groups,
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
        if args.freeze_room:
            model.room.eval()
        running = 0.0
        seen = 0
        for batch in train_loader:
            target, footprint, control = data_batch(
                batch,
                taxonomy,
                device,
            )
            outputs = model(footprint, control_image=control)
            room_loss, components = room_identity_loss(
                outputs["room_logits"],
                target,
                footprint,
                taxonomy.room_ids,
                class_weights=room_weights,
                dice_weight=args.dice_weight,
                boundary_weight=args.boundary_weight,
            )
            object_loss, object_components = structured_object_loss(
                outputs,
                target,
                footprint,
                taxonomy,
                positive_class_weights=positive,
            )
            loss = (
                args.room_loss_weight * room_loss
                + args.object_loss_weight * object_loss
            )
            components.update(object_components)
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
        score = (
            validation["room_macro_iou"]
            + 0.25 * validation["instance_room_f1"]
            + 0.1 * validation["instance_type_accuracy"]
            + 0.05 * validation["wall_iou"]
            + 0.05 * validation["opening_f1"]
        )
        row = {
            "epoch": epoch,
            "train_loss": running / max(1, seen),
            **{key: float(value) for key, value in components.items()},
            **validation,
            "selection_score": score,
        }
        history.append(row)
        with (args.output_dir / "train.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        state = {
            "model": model.state_dict(),
            "config": {
                **vars(args),
                "model": "raster_control_bank",
                "input_protocol": (
                    f"{args.input_rendering}_cubicasa_raster"
                    if args.input_rendering != "nonroom"
                    else "non_room_annotation_rendering"
                ),
            },
            "taxonomy": taxonomy.to_json(),
            "metrics": validation,
        }
        torch.save(state, args.output_dir / "last.pt")
        if score > best_score:
            best_score = score
            torch.save(state, args.output_dir / "best.pt")
        print(json.dumps(row), flush=True)
    report = {
        "status": "complete",
        "scope": "full-semantic raster-to-CAD control screen",
        "train_cases": len(train_dataset),
        "validation_cases": len(val_dataset),
        "best_selection_score": best_score,
        "history": history,
        "config": vars(args),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, default=str))


__all__ = ["data_batch", "evaluate", "main"]


if __name__ == "__main__":
    main()
