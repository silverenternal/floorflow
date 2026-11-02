#!/usr/bin/env python3
"""Train leak-free joint contour-to-program-and-layout prediction."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from floorflow.data.class_balance import load_positive_weights
from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.evaluation.room_metric_suite import room_metric_suite
from floorflow.heads.contour_program_layout import ContourJointLayoutPredictor
from floorflow.heads.room_program import (
    cardinality_gated_program,
    gate_room_labels_to_program,
    room_presence_target,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path(".data/cubicasa5k"))
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("outputs/cache/cubicasa5k_full_v2"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5.0e-4)
    parser.add_argument("--program-weight", type=float, default=0.5)
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument(
        "--program-gating",
        choices=("none", "hard"),
        default="none",
    )
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=191)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def room_loss(logits, labels, room_ids, class_weights=None):
    lookup = torch.full(
        (max(int(labels.max()), int(room_ids.max())) + 1,),
        -100,
        dtype=torch.long,
        device=labels.device,
    )
    lookup[room_ids] = torch.arange(len(room_ids), device=labels.device)
    target = lookup[labels.clamp_max(len(lookup) - 1)]
    ce = F.cross_entropy(
        logits,
        target.clamp_min(0),
        weight=class_weights,
        ignore_index=-100,
    )
    probability = logits.softmax(dim=1)
    one_hot = F.one_hot(
        target.clamp_min(0),
        len(room_ids),
    ).permute(0, 3, 1, 2).float()
    valid = (target >= 0)[:, None].float()
    intersection = (probability * one_hot * valid).sum(dim=(0, 2, 3))
    denominator = (probability * valid).sum(dim=(0, 2, 3)) + (
        one_hot * valid
    ).sum(dim=(0, 2, 3))
    dice = 1.0 - (
        (2.0 * intersection + 1.0) / (denominator + 1.0)
    ).mean()
    return ce + dice


@torch.inference_mode()
def validate(model, loader, taxonomy, device, program_gating):
    model.eval()
    room_ids = torch.tensor(taxonomy.room_ids, device=device)
    rows = []
    for target, footprint, _floor_ids, _names in loader:
        target = target.to(device)
        footprint = footprint.to(device)
        output = model(footprint)
        program = cardinality_gated_program(output["presence_logits"])
        prediction = room_ids[output["layout_logits"].argmax(dim=1)]
        if program_gating == "hard":
            prediction = gate_room_labels_to_program(
                prediction,
                program,
                taxonomy.room_ids,
            )
        prediction = prediction * footprint.long()
        expected_program = room_presence_target(target, taxonomy).bool()
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
            rows.append(
                {
                    "program_exact": float(
                        (program[index] == expected_program[index]).all()
                    ),
                    "room_macro_iou": suite["visible_semantic"]["room_macro_iou"],
                    "instance_geometric_iou": instance["matched_geometric_iou"],
                    "instance_room_f1": instance["instance_f1_at_threshold"],
                    "instance_type_accuracy": instance["matched_type_accuracy"],
                }
            )
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    train = CubiCasaFloorplanDataset(
        args.data_root,
        "train",
        image_size=256,
        cache_root=args.cache_root,
        limit=args.limit_train,
        augment=True,
    )
    val = CubiCasaFloorplanDataset(
        args.data_root,
        "val",
        image_size=256,
        cache_root=args.cache_root,
        limit=args.limit_val,
        augment=False,
    )
    taxonomy = train.taxonomy
    common = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(train, shuffle=True, **common)
    val_loader = DataLoader(val, shuffle=False, **common)
    model = ContourJointLayoutPredictor(
        len(taxonomy.room_ids),
        base_channels=args.base_channels,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1.0e-4)
    room_ids = torch.tensor(taxonomy.room_ids, device=device)
    room_weights = None
    if not args.no_class_weights:
        positive = torch.from_numpy(
            load_positive_weights(args.cache_root, taxonomy.num_target_channels, 20.0)
        ).to(device)
        room_weights = positive[room_ids - 1]
        room_weights = room_weights / room_weights.mean().clamp_min(1.0e-6)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best = -1.0
    for epoch in range(args.epochs):
        started = time.perf_counter()
        model.train()
        totals = {"loss": 0.0, "room": 0.0, "program": 0.0}
        seen = 0
        for target, footprint, _floor_ids, _names in train_loader:
            target = target.to(device)
            footprint = footprint.to(device)
            output = model(footprint)
            room = room_loss(
                output["layout_logits"],
                target,
                room_ids,
                class_weights=room_weights,
            )
            program_target = room_presence_target(target, taxonomy)
            program = F.binary_cross_entropy_with_logits(
                output["presence_logits"],
                program_target,
                pos_weight=room_weights,
            )
            loss = room + args.program_weight * program
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for name, value in (("loss", loss), ("room", room), ("program", program)):
                totals[name] += float(value.detach()) * target.shape[0]
            seen += target.shape[0]
        metrics = validate(
            model,
            val_loader,
            taxonomy,
            device,
            args.program_gating,
        )
        record = {
            "epoch": epoch,
            **{
                f"train_{name}": value / max(1, seen)
                for name, value in totals.items()
            },
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
