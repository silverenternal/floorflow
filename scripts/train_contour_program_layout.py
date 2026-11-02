#!/usr/bin/env python3
"""Train a contour + room-program coarse layout predictor."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from floorflow.data.control_cache import CachedControlDataset
from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.evaluation.room_metric_suite import room_metric_suite
from floorflow.heads.contour_program_layout import ContourProgramLayoutPredictor
from floorflow.heads.room_program import (
    RoomProgramHead,
    gate_room_labels_to_program,
    room_presence_target,
)
from floorflow.heads.room_program_set import (
    RoomProgramSetHead,
    decode_room_program,
    load_room_program_set_state,
)
from floorflow.training.checkpoint import load_checkpoint_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path(".data/cubicasa5k"))
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("outputs/cache/cubicasa5k_full_v2"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/train/contour_program_layout_v1"),
    )
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=191)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--program-source",
        choices=("gt", "predicted"),
        default="gt",
    )
    parser.add_argument("--program-checkpoint", type=Path, default=None)
    parser.add_argument("--program-cache", type=Path, default=None)
    parser.add_argument(
        "--initialize-layout-checkpoint",
        type=Path,
        default=None,
        help="Initialize from a layout teacher before predicted-program adaptation.",
    )
    parser.add_argument(
        "--program-head-type",
        choices=("auto", "bce", "set"),
        default="auto",
    )
    parser.add_argument(
        "--program-gating",
        choices=("none", "hard"),
        default="hard",
    )
    return parser.parse_args()


def room_loss(logits, labels, room_ids):
    lookup = torch.full(
        (max(int(labels.max()), int(room_ids.max())) + 1,),
        -100,
        dtype=torch.long,
        device=labels.device,
    )
    lookup[room_ids] = torch.arange(len(room_ids), device=labels.device)
    target = lookup[labels.clamp_max(len(lookup) - 1)]
    ce = F.cross_entropy(logits, target.clamp_min(0), ignore_index=-100)
    probability = logits.softmax(dim=1)
    one_hot = F.one_hot(
        target.clamp_min(0),
        len(room_ids),
    ).permute(0, 3, 1, 2).float()
    valid = (target >= 0)[:, None].float()
    one_hot = one_hot * valid
    intersection = (probability * one_hot).sum(dim=(0, 2, 3))
    denominator = (probability * valid).sum(dim=(0, 2, 3)) + one_hot.sum(
        dim=(0, 2, 3)
    )
    dice = 1.0 - (
        (2.0 * intersection + 1.0) / (denominator + 1.0)
    ).mean()
    return ce + dice


@torch.inference_mode()
def predict_program(backbone, head, footprint, floor_ids):
    zero = torch.zeros(
        footprint.shape[0],
        backbone.target_channels,
        *footprint.shape[-2:],
        device=footprint.device,
        dtype=footprint.dtype,
    )
    features = backbone._encode_features(
        zero,
        torch.zeros(footprint.shape[0], device=footprint.device),
        footprint,
        floor_ids=floor_ids,
    )
    output = head(features)
    if isinstance(head, RoomProgramSetHead):
        return decode_room_program(output)
    return output["presence_logits"].sigmoid() > 0.5


@torch.inference_mode()
def validate(
    model,
    loader,
    taxonomy,
    room_ids,
    device,
    program_model=None,
    program_gating="hard",
    program_cache=False,
):
    model.eval()
    values = []
    for batch in loader:
        controls = batch[4] if len(batch) == 5 else None
        target, footprint, floor_ids, _names = batch[:4]
        target = target.to(device)
        footprint = footprint.to(device)
        if controls is not None:
            program = controls["program"].to(device)
        elif program_model is None:
            program = room_presence_target(target, taxonomy)
        else:
            program = predict_program(
                program_model[0],
                program_model[1],
                footprint,
                floor_ids.to(device),
            ) > 0.5
        logits = model(footprint, program)
        prediction = room_ids[logits.argmax(dim=1)]
        if program_gating == "hard":
            prediction = gate_room_labels_to_program(
                prediction,
                program,
                taxonomy.room_ids,
            )
        prediction = prediction * footprint.long()
        for index in range(target.shape[0]):
            support = torch.isin(prediction[index], room_ids)
            suite = room_metric_suite(
                prediction[index].cpu().numpy(),
                target[index].cpu().numpy(),
                footprint[index].cpu().numpy(),
                taxonomy,
                predicted_room_support=support.cpu().numpy(),
                predicted_instance_support=support.cpu().numpy(),
                target_room_support=torch.isin(
                    target[index],
                    room_ids,
                ).cpu().numpy(),
            )
            instance = suite["matched_instance"]
            values.append(
                {
                    "room_macro_iou": suite["visible_semantic"]["room_macro_iou"],
                    "instance_geometric_iou": instance["matched_geometric_iou"],
                    "instance_room_f1": instance["instance_f1_at_threshold"],
                    "instance_type_accuracy": instance["matched_type_accuracy"],
                }
            )
    return {
        key: float(np.mean([row[key] for row in values]))
        for key in values[0]
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    common = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    train = CubiCasaFloorplanDataset(
        args.data_root,
        "train",
        image_size=args.image_size,
        cache_root=args.cache_root,
        limit=args.limit_train,
        augment=True,
    )
    val = CubiCasaFloorplanDataset(
        args.data_root,
        "val",
        image_size=args.image_size,
        cache_root=args.cache_root,
        limit=args.limit_val,
        augment=False,
    )
    taxonomy = train.taxonomy
    if args.program_cache is not None:
        train = CachedControlDataset(train, args.program_cache, "train")
        val = CachedControlDataset(val, args.program_cache, "val")
    room_ids = torch.tensor(taxonomy.room_ids, device=device)
    train_loader = DataLoader(train, shuffle=True, **common)
    val_loader = DataLoader(val, shuffle=False, **common)
    model = ContourProgramLayoutPredictor(
        len(taxonomy.room_ids),
        base_channels=args.base_channels,
    ).to(device)
    if args.initialize_layout_checkpoint is not None:
        layout_payload = torch.load(
            args.initialize_layout_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        model.load_state_dict(layout_payload["model"], strict=False)
    program_model = None
    if args.program_source == "predicted":
        if args.program_checkpoint is None:
            raise ValueError("predicted program requires --program-checkpoint")
        payload = torch.load(
            args.program_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        backbone_path = payload.get("backbone_checkpoint")
        if backbone_path is None:
            raise ValueError("program checkpoint has no backbone_checkpoint")
        backbone, _config, _taxonomy, _ = load_checkpoint_model(
            Path(backbone_path),
            device,
            taxonomy=taxonomy,
        )
        program_head_type = args.program_head_type
        if program_head_type == "auto":
            program_head_type = str(
                payload.get("head_type", "set" if "queries" in payload.get("config", {}) else "bce")
            )
        if program_head_type == "set":
            program_config = payload.get("config", {})
            head = RoomProgramSetHead(
                backbone.output.in_channels,
                len(taxonomy.room_ids),
                hidden_dim=int(program_config.get("hidden_dim", 96)),
                tokens=int(program_config.get("tokens", 8)),
                layers=int(program_config.get("layers", 1)),
                heads=int(program_config.get("heads", 4)),
                num_queries=int(program_config.get("queries", 32)),
            ).to(device)
        else:
            head = RoomProgramHead(
                backbone.output.in_channels,
                len(taxonomy.room_ids),
            ).to(device)
        if isinstance(head, RoomProgramSetHead):
            load_room_program_set_state(head, payload["head"])
        else:
            head.load_state_dict(payload["head"])
        backbone.eval().requires_grad_(False)
        head.eval().requires_grad_(False)
        program_model = (backbone, head)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1.0e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs),
        eta_min=args.lr * 0.05,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best = -1.0
    for epoch in range(args.epochs):
        started = time.perf_counter()
        model.train()
        total = 0.0
        seen = 0
        for batch in train_loader:
            controls = batch[4] if len(batch) == 5 else None
            target, footprint, _floor_ids, _names = batch[:4]
            target = target.to(device)
            footprint = footprint.to(device)
            if controls is not None:
                program = controls["program"].to(device)
            elif program_model is None:
                program = room_presence_target(target, taxonomy)
            else:
                program = predict_program(
                    program_model[0],
                    program_model[1],
                    footprint,
                    _floor_ids.to(device),
                )
            logits = model(footprint, program)
            loss = room_loss(logits, target, room_ids)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * target.shape[0]
            seen += target.shape[0]
        scheduler.step()
        metrics = validate(
            model,
            val_loader,
            taxonomy,
            room_ids,
            device,
            program_model=program_model,
            program_gating=args.program_gating,
            program_cache=args.program_cache is not None,
        )
        record = {
            "epoch": epoch,
            "train_loss": total / max(1, seen),
            "seconds": time.perf_counter() - started,
            **metrics,
        }
        print(json.dumps(record), flush=True)
        with (args.output_dir / "train.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        checkpoint = {
            "model": model.state_dict(),
            "taxonomy": taxonomy.to_json(),
            "config": {**vars(args), "num_room_classes": len(taxonomy.room_ids)},
            "metrics": metrics,
        }
        torch.save(checkpoint, args.output_dir / "last.pt")
        if metrics["room_macro_iou"] > best:
            best = metrics["room_macro_iou"]
            torch.save(checkpoint, args.output_dir / "best.pt")


if __name__ == "__main__":
    main()
