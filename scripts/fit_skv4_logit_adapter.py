#!/usr/bin/env python3
"""Fit a class-bias/temperature adapter on skv4 train and select on val."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from floorflow.data.skv4_dataset import Skv4PlanDataset
from floorflow.evaluation.room_metric_suite import room_metric_suite
from floorflow.training.checkpoint import load_checkpoint_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--skv4-root", type=Path, default=Path("outputs/cache/skv4"))
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("outputs/cache/cubicasa5k_full_v2"),
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--bias-l2", type=float, default=0.01)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def room_target(target: torch.Tensor, room_ids: torch.Tensor) -> torch.Tensor:
    output = torch.full_like(target, -100)
    for index, class_id in enumerate(room_ids.tolist()):
        output[target == class_id] = index
    return output


@torch.inference_mode()
def evaluate_adapter(
    model,
    loader,
    taxonomy,
    room_ids,
    bias,
    temperature,
    device,
) -> dict[str, float]:
    model.eval()
    rows = []
    for target, footprint, _floor, _name in loader:
        target = target.to(device)
        footprint = footprint[:, None].to(device)
        zeros = torch.zeros(
            target.shape[0],
            model.target_channels,
            *target.shape[-2:],
            device=device,
        )
        _velocity, auxiliary = model.forward_with_aux(
            zeros,
            torch.zeros(target.shape[0], device=device),
            footprint,
        )
        logits = (
            auxiliary["room_partition_logits"] + bias[None, :, None, None]
        ) / temperature.clamp_min(0.05)
        prediction = room_ids[logits.argmax(dim=1)] * footprint[:, 0].long()
        for index in range(target.shape[0]):
            support = torch.isin(prediction[index], room_ids)
            suite = room_metric_suite(
                prediction[index].cpu().numpy(),
                target[index].cpu().numpy(),
                footprint[index, 0].cpu().numpy(),
                taxonomy,
                predicted_room_support=support.cpu().numpy(),
                predicted_instance_support=support.cpu().numpy(),
                target_room_support=torch.isin(
                    target[index], room_ids
                ).cpu().numpy(),
            )
            instance = suite["matched_instance"]
            rows.append(
                {
                    "room_macro_iou": suite["visible_semantic"]["room_macro_iou"],
                    "instance_geometric_iou": instance["matched_geometric_iou"],
                    "instance_room_f1": instance["instance_f1_at_threshold"],
                    "instance_type_accuracy": instance["matched_type_accuracy"],
                    "class_constrained_iou": instance[
                        "matched_class_constrained_iou"
                    ],
                }
            )
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model, _config, taxonomy, _state = load_checkpoint_model(
        args.checkpoint,
        device,
        taxonomy=None,
    )
    room_ids = torch.tensor(taxonomy.room_ids, device=device)
    train = Skv4PlanDataset(
        args.skv4_root,
        "train",
        taxonomy,
        augment=True,
        limit=args.limit_train,
    )
    val = Skv4PlanDataset(
        args.skv4_root,
        "val",
        taxonomy,
        augment=False,
        limit=args.limit_val,
    )
    loader_options = {
        "num_workers": args.workers,
        "persistent_workers": args.workers > 0,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        train,
        batch_size=args.batch_size,
        shuffle=True,
        **loader_options,
    )
    val_loader = DataLoader(
        val,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_options,
    )
    bias = torch.zeros(len(taxonomy.room_ids), device=device, requires_grad=True)
    log_temperature = torch.zeros((), device=device, requires_grad=True)
    optimizer = torch.optim.Adam([bias, log_temperature], lr=args.lr)
    best = None
    best_state = None
    for epoch in range(args.epochs):
        model.eval()
        for target, footprint, _floor, _name in train_loader:
            target = target.to(device)
            footprint = footprint[:, None].to(device)
            zeros = torch.zeros(
                target.shape[0],
                model.target_channels,
                *target.shape[-2:],
                device=device,
            )
            with torch.no_grad():
                _velocity, auxiliary = model.forward_with_aux(
                    zeros,
                    torch.zeros(target.shape[0], device=device),
                    footprint,
                )
                base_logits = auxiliary["room_partition_logits"].float()
                labels = room_target(target, room_ids)
                valid = (labels >= 0) & footprint[:, 0].bool()
            temperature = log_temperature.exp().clamp(0.25, 4.0)
            adapted = (base_logits + bias[None, :, None, None]) / temperature
            loss = F.cross_entropy(
                adapted.permute(0, 2, 3, 1)[valid],
                labels[valid],
            ) + args.bias_l2 * bias.square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                bias.clamp_(-2.0, 2.0)
        metrics = evaluate_adapter(
            model,
            val_loader,
            taxonomy,
            room_ids,
            bias,
            log_temperature.exp().clamp(0.25, 4.0),
            device,
        )
        print(
            json.dumps({"epoch": epoch, **metrics}),
            flush=True,
        )
        if best is None or metrics["room_macro_iou"] > best["room_macro_iou"]:
            best = metrics
            best_state = {
                "bias": bias.detach().cpu().tolist(),
                "temperature": float(
                    log_temperature.exp().clamp(0.25, 4.0).detach().cpu()
                ),
                "epoch": epoch,
                "val": metrics,
                "class_ids": list(taxonomy.room_ids),
                "best_of_n_used": False,
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(best_state, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(best_state, indent=2))


if __name__ == "__main__":
    main()
