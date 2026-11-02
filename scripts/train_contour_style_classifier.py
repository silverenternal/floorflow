#!/usr/bin/env python3
"""Train a small contour-only CNN for style-bank classification."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.heads.contour_style_classifier import ContourStyleClassifier
from floorflow.representation.core import (
    floorplan_feature_vector,
    transform_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--style-bank", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path(".data/cubicasa5k"))
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("outputs/cache/cubicasa5k_full_v2"),
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def nearest_style_labels(dataset, bank) -> np.ndarray:
    center = np.asarray(bank["feature_center"], dtype=np.float64)
    scale = np.asarray(bank["feature_scale"], dtype=np.float64)
    centroids = np.asarray(bank["centroids_standardized"], dtype=np.float64)
    labels = []
    for index in range(len(dataset)):
        target, _footprint, _floor, _name = dataset[index]
        feature = floorplan_feature_vector(target.numpy(), dataset.taxonomy)
        standardized = transform_features(feature[None], center, scale)
        labels.append(
            int(np.argmin(((centroids - standardized) ** 2).sum(axis=1)))
        )
    return np.asarray(labels, dtype=np.int64)


@torch.inference_mode()
def evaluate(model, loader, device) -> float:
    model.eval()
    correct = 0
    total = 0
    for footprint, label in loader:
        logits = model(footprint.to(device, non_blocking=True))
        correct += int((logits.argmax(dim=1) == label.to(device)).sum())
        total += label.numel()
    return correct / max(1, total)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    bank = json.loads(args.style_bank.read_text(encoding="utf-8"))
    train = CubiCasaFloorplanDataset(
        args.data_root,
        "train",
        image_size=256,
        cache_root=args.cache_root,
        augment=False,
    )
    val = CubiCasaFloorplanDataset(
        args.data_root,
        "val",
        image_size=256,
        cache_root=args.cache_root,
        augment=False,
    )
    train_names = bank["assignments"]["train"]
    train_labels = np.asarray(
        [train_names[str(train[index][3])] for index in range(len(train))],
        dtype=np.int64,
    )
    val_labels = nearest_style_labels(val, bank)
    train_tensors = torch.utils.data.TensorDataset(
        torch.stack([train[index][1] for index in range(len(train))]),
        torch.from_numpy(train_labels),
    )
    val_tensors = torch.utils.data.TensorDataset(
        torch.stack([val[index][1] for index in range(len(val))]),
        torch.from_numpy(val_labels),
    )
    train_loader = DataLoader(
        train_tensors,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_tensors,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    model = ContourStyleClassifier(bank["num_styles"]).to(device)
    counts = Counter(train_labels.tolist())
    weights = torch.tensor(
        [
            len(train_labels) / (len(counts) * counts[index])
            for index in range(bank["num_styles"])
        ],
        device=device,
        dtype=torch.float32,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    best = -1.0
    best_epoch = -1
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        seen = 0
        for footprint, label in train_loader:
            footprint = footprint.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            logits = model(footprint)
            loss = torch.nn.functional.cross_entropy(
                logits,
                label,
                weight=weights,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * label.numel()
            seen += label.numel()
        scheduler.step()
        accuracy = evaluate(model, val_loader, device)
        majority = max(Counter(val_labels.tolist()).values()) / len(val_labels)
        record = {
            "epoch": epoch,
            "train_loss": total / max(1, seen),
            "val_style_accuracy": accuracy,
            "val_majority_accuracy": majority,
        }
        print(json.dumps(record), flush=True)
        if accuracy > best:
            best = accuracy
            best_epoch = epoch
            torch.save(
                {
                    "model": model.state_dict(),
                    "num_styles": int(bank["num_styles"]),
                    "style_bank": str(args.style_bank),
                    "config": vars(args),
                    "metrics": record,
                    "input_mode": "contour_only",
                },
                args.output,
            )
    print(
        json.dumps(
            {
                "best_epoch": best_epoch,
                "best_val_style_accuracy": best,
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
