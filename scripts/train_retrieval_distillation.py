#!/usr/bin/env python3
"""Leave-one-out train-only retrieval distillation for the room pixel head."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.distribution.checkpoint import (
    restore_distribution_model,
    save_distribution_model,
)
from floorflow.objectives.condition_objective import room_macro_iou_from_labels
from floorflow.representation.contour_memory import ContourPartitionMemory
from floorflow.training.retrieval_distillation import (
    distillation_loss,
    excluded_indices,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path(".data/cubicasa5k"))
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("outputs/cache/cubicasa5k_full_v2"),
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit-train", type=int, default=1024)
    parser.add_argument("--limit-val", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--preserve-weight", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def student_logits(model, footprint, floor_ids, style_ids=None):
    distribution = model.contour_initial_state(
        footprint,
        floor_ids=floor_ids,
        style_ids=style_ids,
    )
    return distribution["pixel"]["room_logits"]


@torch.inference_mode()
def evaluate(model, memory, loader, taxonomy, path_to_index, device):
    student_values = []
    teacher_values = []
    room_ids = torch.tensor(taxonomy.room_ids, device=device)
    for target, footprint, floor_ids, names in loader:
        target = target.to(device)
        footprint = footprint[:, None].to(device)
        floor_ids = floor_ids.to(device)
        style_ids = torch.zeros(
            footprint.shape[0],
            dtype=torch.long,
            device=device,
        )
        logits = student_logits(model, footprint, floor_ids, style_ids)
        student = room_ids[logits.argmax(dim=1)] * footprint[:, 0].long()
        teacher_logits = memory.retrieve_logits(
            footprint[:, 0],
            taxonomy,
            device,
            top_k=1,
        )
        teacher = room_ids[teacher_logits.argmax(dim=1)] * footprint[:, 0].long()
        for index in range(target.shape[0]):
            student_values.append(
                room_macro_iou_from_labels(
                    student[index : index + 1],
                    target[index : index + 1],
                    taxonomy,
                )
            )
            teacher_values.append(
                room_macro_iou_from_labels(
                    teacher[index : index + 1],
                    target[index : index + 1],
                    taxonomy,
                )
            )
    return {
        "student_room_macro_iou": float(np.mean(student_values)),
        "retrieval_teacher_room_macro_iou": float(np.mean(teacher_values)),
        "cases": len(student_values),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model, taxonomy, _payload = restore_distribution_model(
        args.checkpoint,
        device,
    )
    reference_model, _reference_taxonomy, _reference_payload = (
        restore_distribution_model(args.checkpoint, device)
    )
    reference_model.eval().requires_grad_(False)
    train = CubiCasaFloorplanDataset(
        args.data_root,
        "train",
        image_size=256,
        cache_root=args.cache_root,
        augment=True,
        limit=args.limit_train,
    )
    val = CubiCasaFloorplanDataset(
        args.data_root,
        "val",
        image_size=256,
        cache_root=args.cache_root,
        augment=False,
        limit=args.limit_val,
    )
    memory = ContourPartitionMemory.from_cache(args.cache_root, top_k=args.top_k)
    actual_top_k = min(args.top_k, len(memory) - 1)
    memory.top_k = actual_top_k
    path_to_index = {
        path.relative_to(args.cache_root / "train"): index
        for index, path in enumerate(memory.paths)
    }
    common = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train, shuffle=True, **common)
    val_loader = DataLoader(val, shuffle=False, **common)
    parameters = [
        parameter
        for name, parameter in model.hub.pixel.named_parameters()
        if "style" not in name.lower() and parameter.requires_grad
    ]
    if not parameters:
        raise ValueError("distillation found no non-style pixel parameters")
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=1.0e-4)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best = -1.0
    for epoch in range(args.epochs):
        started = time.perf_counter()
        model.train()
        total = 0.0
        seen = 0
        for step, (_target, footprint, floor_ids, names) in enumerate(train_loader):
            footprint = footprint[:, None].to(device, non_blocking=True)
            floor_ids = floor_ids.to(device, non_blocking=True)
            teacher = memory.retrieve_logits(
                footprint[:, 0],
                taxonomy,
                device,
                top_k=actual_top_k,
                exclude_indices=excluded_indices(names, path_to_index),
            )
            style_ids = (
                torch.arange(footprint.shape[0], device=device)
                + step * footprint.shape[0]
            ) % max(1, model.num_styles)
            student = student_logits(model, footprint, floor_ids, style_ids)
            loss = distillation_loss(student, teacher, args.temperature)
            with torch.inference_mode():
                reference = student_logits(
                    reference_model,
                    footprint,
                    floor_ids,
                    style_ids,
                )
            loss = loss + float(args.preserve_weight) * distillation_loss(
                student,
                reference,
                1.0,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            total += float(loss.detach()) * footprint.shape[0]
            seen += footprint.shape[0]
        metrics = evaluate(
            model,
            memory,
            val_loader,
            taxonomy,
            path_to_index,
            device,
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
        save_distribution_model(
            model,
            args.output_dir / "last.pt",
            taxonomy=taxonomy,
            extra={"training_args": vars(args), "distillation": record},
        )
        if metrics["student_room_macro_iou"] > best:
            best = metrics["student_room_macro_iou"]
            save_distribution_model(
                model,
                args.output_dir / "best.pt",
                taxonomy=taxonomy,
                extra={"training_args": vars(args), "distillation": record},
            )


if __name__ == "__main__":
    main()
