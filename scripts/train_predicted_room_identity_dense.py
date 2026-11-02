#!/usr/bin/env python3
"""Train a dense residual room-identity corrector on predicted controls."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from train_predicted_room_identity import identity_features

from floorflow.data.class_balance import load_positive_weights
from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.distribution.checkpoint import restore_distribution_model
from floorflow.evaluation.room_metric_suite import room_metric_suite
from floorflow.heads.predicted_room_identity import (
    PredictedRoomIdentityHead,
    room_identity_loss,
)
from floorflow.inference.room_segmenter_prior import load_room_segmenter_prior

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--distribution-checkpoint",
        type=Path,
        default=ROOT
        / "outputs/train/distribution_dmd_adversarial_datamap4_v11/best.pt",
    )
    parser.add_argument(
        "--room-segmenter-checkpoint",
        type=Path,
        default=ROOT
        / "outputs/train/room_segmenter_r18_uniform_full40_v2/best.pt",
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
        default=ROOT / "outputs/train/predicted_room_identity_dense_v1",
    )
    parser.add_argument("--limit-train", type=int, default=1024)
    parser.add_argument("--limit-val", type=int, default=491)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--hidden-channels", type=int, default=96)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--dice-weight", type=float, default=0.5)
    parser.add_argument("--boundary-weight", type=float, default=0.5)
    parser.add_argument("--class-balance", action="store_true")
    parser.add_argument("--use-image", action="store_true")
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2902)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def evaluate(
    head,
    distribution,
    segmenter,
    loader,
    taxonomy,
    room_ids: tuple[int, ...],
    device: torch.device,
) -> dict[str, float]:
    rows = []
    room_tensor = torch.tensor(room_ids, device=device)
    for target, footprint, floor_ids, _names, *extra in loader:
        target = target.to(device)
        footprint = footprint.to(device)
        floor_ids = floor_ids.to(device)
        features, base_logits = identity_features(
            distribution,
            segmenter,
            footprint,
            floor_ids,
            room_ids,
        )
        image = extra[0].to(device) if extra else None
        logits = head(features, base_logits, image=image)["room_logits"]
        labels = room_tensor[logits.argmax(dim=1)] * footprint.long()
        for index in range(target.shape[0]):
            room_support = torch.isin(labels[index], room_tensor)
            target_support = torch.isin(target[index], room_tensor)
            suite = room_metric_suite(
                labels[index].cpu().numpy(),
                target[index].cpu().numpy(),
                footprint[index].cpu().numpy(),
                taxonomy,
                predicted_room_support=room_support.cpu().numpy(),
                predicted_instance_support=room_support.cpu().numpy(),
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


def selection_score(metrics: dict[str, float]) -> float:
    return (
        float(metrics["room_macro_iou"])
        + 0.25 * float(metrics["instance_room_f1"])
        + 0.1 * float(metrics["instance_type_accuracy"])
    )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    distribution, taxonomy, _ = restore_distribution_model(
        args.distribution_checkpoint,
        device,
    )
    distribution.eval().requires_grad_(False)
    segmenter, _ = load_room_segmenter_prior(
        args.room_segmenter_checkpoint,
        taxonomy,
        device,
    )
    segmenter.eval().requires_grad_(False)
    room_ids = tuple(distribution.hub.hierarchy.room_ids)
    feature_channels = 2 * len(room_ids) + 2
    head = PredictedRoomIdentityHead(
        feature_channels,
        len(room_ids),
        hidden_channels=args.hidden_channels,
        blocks=args.blocks,
        use_image=args.use_image,
    ).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
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
    train_dataset = CubiCasaFloorplanDataset(
        args.data_root,
        "train",
        image_size=256,
        cache_root=args.cache_root,
        augment=True,
        limit=args.limit_train,
        include_image=args.use_image,
    )
    val_dataset = CubiCasaFloorplanDataset(
        args.data_root,
        "val",
        image_size=256,
        cache_root=args.cache_root,
        augment=False,
        limit=args.limit_val,
        include_image=args.use_image,
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
        head.train()
        running = 0.0
        seen = 0
        for target, footprint, floor_ids, _names, *extra in train_loader:
            target = target.to(device)
            footprint = footprint.to(device)
            floor_ids = floor_ids.to(device)
            image = extra[0].to(device) if extra else None
            features, base_logits = identity_features(
                distribution,
                segmenter,
                footprint,
                floor_ids,
                room_ids,
            )
            outputs = head(features, base_logits, image=image)
            loss, components = room_identity_loss(
                outputs["room_logits"],
                target,
                footprint,
                room_ids,
                class_weights=class_weights,
                dice_weight=args.dice_weight,
                boundary_weight=args.boundary_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                head.parameters(),
                args.gradient_clip,
            )
            optimizer.step()
            running += float(loss.detach()) * target.shape[0]
            seen += target.shape[0]
        validation = evaluate(
            head,
            distribution,
            segmenter,
            val_loader,
            taxonomy,
            room_ids,
            device,
        )
        row = {
            "epoch": epoch,
            "train_loss": running / max(1, seen),
            **{
                key: float(value)
                for key, value in components.items()
            },
            **validation,
            "residual_scale": float(head.residual_scale.detach()),
            "selection_score": selection_score(validation),
        }
        history.append(row)
        with (args.output_dir / "train.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        if row["selection_score"] > best_score:
            best_score = row["selection_score"]
            torch.save(
                {
                    "model": head.state_dict(),
                    "config": {
                        **vars(args),
                        "feature_channels": feature_channels,
                        "room_ids": room_ids,
                    },
                    "taxonomy": taxonomy.to_json(),
                    "metrics": validation,
                },
                args.output_dir / "best.pt",
            )
        torch.save(
            {
                "model": head.state_dict(),
                "config": {
                    **vars(args),
                    "feature_channels": feature_channels,
                    "room_ids": room_ids,
                },
                "taxonomy": taxonomy.to_json(),
                "metrics": validation,
            },
            args.output_dir / "last.pt",
        )
        print(json.dumps(row), flush=True)
    report = {
        "status": "complete",
        "target_used_at_inference": False,
        "condition_mode": "predicted_control",
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


__all__ = ["evaluate", "main", "selection_score"]


if __name__ == "__main__":
    main()
