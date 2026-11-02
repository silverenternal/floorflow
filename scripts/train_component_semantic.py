#!/usr/bin/env python3
"""Train a component-only room semantic classifier on frozen room support."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.heads.component_semantic import (
    ComponentSemanticClassifier,
    component_records,
)
from floorflow.objectives.condition_objective import partition_room_labels
from floorflow.representation.contour_memory import ContourPartitionMemory
from floorflow.training.checkpoint import load_checkpoint_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path(".data/cubicasa5k"))
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("outputs/cache/cubicasa5k_full_v2"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--retrieval-prior-weight", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--feature-batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=191)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def memory_indices(names, memory, cache_root: Path) -> np.ndarray | None:
    lookup = {
        path.resolve(): index
        for index, path in enumerate(memory.paths)
    }
    indices = []
    for name in names:
        relative = Path(str(name).replace("__", "/"))
        path = (
            Path(cache_root)
            / "train"
            / relative.parent
            / f"{relative.name.replace('-', '_')}.npz"
        ).resolve()
        if path not in lookup:
            return None
        indices.append(lookup[path])
    return np.asarray(indices, dtype=np.int64)


@torch.inference_mode()
def extract_records(
    model,
    loader,
    taxonomy,
    memory,
    prior_weight,
    device,
    *,
    exclude_self: bool,
    cache_root: Path,
):
    features = []
    targets = []
    for target, footprint, floor_ids, names in loader:
        target = target.to(device)
        footprint = footprint.to(device)
        zeros = torch.zeros(
            target.shape[0],
            model.target_channels,
            *target.shape[-2:],
            device=device,
        )
        _velocity, aux = model.forward_with_aux(
            zeros,
            torch.zeros(target.shape[0], device=device),
            footprint,
            floor_ids=floor_ids.to(device),
        )
        room_logits = aux["room_partition_logits"]
        excluded = (
            memory_indices(names, memory, cache_root)
            if exclude_self
            else None
        )
        if exclude_self and excluded is None:
            raise ValueError("could not map train samples to retrieval memory")
        retrieval = memory.retrieve_logits(
            footprint.cpu().numpy(),
            taxonomy,
            device,
            exclude_indices=excluded,
        )
        support_logits = room_logits + float(prior_weight) * retrieval
        labels = (
            partition_room_labels(support_logits, taxonomy)
            * footprint.long()
        )
        model_probability = room_logits.softmax(dim=1).cpu().numpy()
        retrieval_probability = retrieval.softmax(dim=1).cpu().numpy()
        labels_np = labels.cpu().numpy()
        target_np = target.cpu().numpy()
        footprint_np = footprint.cpu().numpy()
        for index in range(target.shape[0]):
            batch_features, batch_targets, _ids = component_records(
                labels_np[index],
                target_np[index],
                model_probability[index],
                retrieval_probability[index],
                footprint_np[index],
                taxonomy.room_ids,
            )
            features.append(batch_features)
            targets.append(batch_targets)
    return np.concatenate(features), np.concatenate(targets)


@torch.inference_mode()
def evaluate_components(model, features, targets, taxonomy, device):
    model.eval()
    logits = []
    for start in range(0, len(features), 1024):
        logits.append(
            model(
                torch.from_numpy(features[start : start + 1024]).to(device)
            )
        )
    prediction = torch.cat(logits).argmax(dim=-1).cpu().numpy()
    accuracy = float((prediction == targets).mean()) if len(targets) else 0.0
    recalls = []
    for class_index in range(len(taxonomy.room_ids)):
        selected = targets == class_index
        if selected.any():
            recalls.append(float((prediction[selected] == class_index).mean()))
    return {
        "components": len(targets),
        "accuracy": accuracy,
        "macro_recall": float(np.mean(recalls)) if recalls else 0.0,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model, config, taxonomy, _ = load_checkpoint_model(
        args.checkpoint,
        device,
    )
    model.eval().requires_grad_(False)
    memory = ContourPartitionMemory.from_cache(args.cache_root, top_k=1)
    common = {
        "batch_size": args.feature_batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    train_dataset = CubiCasaFloorplanDataset(
        args.data_root,
        "train",
        image_size=int(config.get("image_size", 256)),
        cache_root=args.cache_root,
        limit=args.limit_train,
        augment=False,
    )
    val_dataset = CubiCasaFloorplanDataset(
        args.data_root,
        "val",
        image_size=int(config.get("image_size", 256)),
        cache_root=args.cache_root,
        limit=args.limit_val,
        augment=False,
    )
    train_loader = DataLoader(train_dataset, shuffle=False, **common)
    val_loader = DataLoader(val_dataset, shuffle=False, **common)
    train_features, train_targets = extract_records(
        model,
        train_loader,
        taxonomy,
        memory,
        args.retrieval_prior_weight,
        device,
        exclude_self=True,
        cache_root=args.cache_root,
    )
    val_features, val_targets = extract_records(
        model,
        val_loader,
        taxonomy,
        memory,
        args.retrieval_prior_weight,
        device,
        exclude_self=False,
        cache_root=args.cache_root,
    )
    counts = np.bincount(
        train_targets,
        minlength=len(taxonomy.room_ids),
    )
    class_weights = torch.from_numpy(
        (1.0 / np.sqrt(np.maximum(counts, 1))).astype(np.float32)
    ).to(device)
    class_weights /= class_weights.mean().clamp_min(1.0e-6)
    classifier = ComponentSemanticClassifier(
        train_features.shape[1],
        len(taxonomy.room_ids),
        hidden_size=args.hidden_size,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        classifier.parameters(),
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
        classifier.train()
        permutation = np.random.default_rng(epoch).permutation(
            len(train_features)
        )
        total = 0.0
        for start in range(0, len(train_features), args.batch_size):
            index = permutation[start : start + args.batch_size]
            features = torch.from_numpy(train_features[index]).to(device)
            targets = torch.from_numpy(train_targets[index]).to(device)
            logits = classifier(features)
            loss = F.cross_entropy(
                logits,
                targets,
                weight=class_weights,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(index)
        scheduler.step()
        metrics = evaluate_components(
            classifier,
            val_features,
            val_targets,
            taxonomy,
            device,
        )
        record = {
            "epoch": epoch,
            "train_loss": total / max(1, len(train_features)),
            **metrics,
        }
        print(json.dumps(record), flush=True)
        if metrics["accuracy"] > best:
            best = metrics["accuracy"]
            torch.save(
                {
                    "model": classifier.state_dict(),
                    "feature_size": int(train_features.shape[1]),
                    "num_classes": len(taxonomy.room_ids),
                    "class_ids": list(taxonomy.room_ids),
                    "config": vars(args),
                    "metrics": metrics,
                },
                args.output_dir / "best.pt",
            )


if __name__ == "__main__":
    main()
