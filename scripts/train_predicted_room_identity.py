#!/usr/bin/env python3
"""Train a small room-identity head over frozen target-free control features."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.distribution.checkpoint import restore_distribution_model
from floorflow.evaluation.room_metric_suite import room_metric_suite
from floorflow.heads.room_mask_decoder import (
    RoomMaskDecoder,
    room_mask_loss,
    room_mask_partition_loss,
    room_mask_targets,
)
from floorflow.heads.room_mask_fusion import fused_room_mask_logits
from floorflow.inference.room_segmenter_prior import (
    load_room_segmenter_prior,
    predict_room_segmenter_prior,
)
from floorflow.representation.core import extract_room_instances

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
        default=ROOT / "outputs/train/predicted_room_identity_v1",
    )
    parser.add_argument("--limit-train", type=int, default=1024)
    parser.add_argument("--limit-val", type=int, default=191)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--queries", type=int, default=32)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--boundary-weight", type=float, default=2.0)
    parser.add_argument("--assignment-class-weight", type=float, default=0.5)
    parser.add_argument("--assignment-mask-weight", type=float, default=2.0)
    parser.add_argument("--class-loss-weight", type=float, default=1.0)
    parser.add_argument("--mask-loss-weight", type=float, default=2.0)
    parser.add_argument("--partition-weight", type=float, default=0.5)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2902)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def boundary_band(footprint: torch.Tensor, radius: int = 4) -> torch.Tensor:
    width = 2 * int(radius) + 1
    eroded = -F.max_pool2d(
        -footprint[:, None].float(),
        width,
        stride=1,
        padding=radius,
    )
    return footprint[:, None].float() - eroded


@torch.no_grad()
def identity_features(
    distribution,
    segmenter,
    footprint: torch.Tensor,
    floor_ids: torch.Tensor,
    room_ids: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    segmenter_logits = predict_room_segmenter_prior(segmenter, footprint)
    state = distribution.contour_initial_state(
        footprint,
        floor_ids=floor_ids,
    )
    dmd_logits = state["pixel"]["logits"][:, room_ids]
    if dmd_logits.shape[-2:] != footprint.shape[-2:]:
        dmd_logits = F.interpolate(
            dmd_logits,
            size=footprint.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    segmenter_probability = segmenter_logits.log_softmax(dim=1)
    dmd_probability = dmd_logits.log_softmax(dim=1)
    features = torch.cat(
        [
            segmenter_probability,
            dmd_probability,
            footprint[:, None].float(),
            boundary_band(footprint),
        ],
        dim=1,
    )
    return features, segmenter_logits


def target_counts(labels: torch.Tensor, taxonomy) -> list[int]:
    return [
        len(
            extract_room_instances(
                row.detach().cpu().numpy(),
                taxonomy,
                min_area=8,
            )
        )
        for row in labels
    ]


@torch.inference_mode()
def evaluate(
    decoder,
    distribution,
    segmenter,
    loader,
    taxonomy,
    room_ids: tuple[int, ...],
    device: torch.device,
) -> dict[str, float]:
    rows = []
    decoder.eval()
    for target, footprint, floor_ids, _names in loader:
        target = target.to(device)
        footprint = footprint.to(device)
        floor_ids = floor_ids.to(device)
        features, segmenter_logits = identity_features(
            distribution,
            segmenter,
            footprint,
            floor_ids,
            room_ids,
        )
        predictions = decoder(
            features,
            prototype_logits=segmenter_logits[:, None],
        )
        logits = fused_room_mask_logits(
            predictions,
            len(taxonomy.room_ids),
        )
        room_tensor = torch.tensor(taxonomy.room_ids, device=device)
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
    decoder = RoomMaskDecoder(
        feature_channels,
        len(taxonomy.room_ids),
        hidden_dim=args.hidden_dim,
        num_queries=args.queries,
        layers=args.layers,
        heads=args.heads,
    ).to(device)
    optimizer = torch.optim.AdamW(
        decoder.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    train_dataset = CubiCasaFloorplanDataset(
        args.data_root,
        "train",
        image_size=256,
        cache_root=args.cache_root,
        augment=True,
        limit=args.limit_train,
    )
    val_dataset = CubiCasaFloorplanDataset(
        args.data_root,
        "val",
        image_size=256,
        cache_root=args.cache_root,
        augment=False,
        limit=args.limit_val,
    )
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": True,
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    history = []
    best_score = -1.0
    for epoch in range(args.epochs):
        decoder.train()
        running = 0.0
        seen = 0
        for target, footprint, floor_ids, _names in train_loader:
            target = target.to(device)
            footprint = footprint.to(device)
            floor_ids = floor_ids.to(device)
            features, segmenter_logits = identity_features(
                distribution,
                segmenter,
                footprint,
                floor_ids,
                room_ids,
            )
            predictions = decoder(
                features,
                prototype_logits=segmenter_logits[:, None],
            )
            target_classes, target_masks, target_valid = room_mask_targets(
                target,
                taxonomy,
                max_rooms=args.queries,
            )
            loss, components = room_mask_loss(
                predictions,
                target_classes,
                target_masks,
                target_valid,
                boundary_weight=args.boundary_weight,
                assignment_class_weight=args.assignment_class_weight,
                assignment_mask_weight=args.assignment_mask_weight,
                class_loss_weight=args.class_loss_weight,
                mask_loss_weight=args.mask_loss_weight,
            )
            partition_loss, partition_components = room_mask_partition_loss(
                predictions,
                target,
                footprint,
                taxonomy,
            )
            loss = loss + args.partition_weight * partition_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                decoder.parameters(),
                args.gradient_clip,
            )
            optimizer.step()
            running += float(loss.detach()) * target.shape[0]
            seen += target.shape[0]
        validation = evaluate(
            decoder,
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
            "mask_loss": float(components["room_mask_loss"]),
            "partition_loss": float(
                partition_components["room_mask_partition_loss"]
            ),
            **validation,
            "selection_score": selection_score(validation),
        }
        history.append(row)
        with (args.output_dir / "train.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        if row["selection_score"] > best_score:
            best_score = row["selection_score"]
            torch.save(
                {
                    "model": decoder.state_dict(),
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
                "model": decoder.state_dict(),
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
        json.dumps(
            report,
            indent=2,
            default=lambda value: str(value)
            if isinstance(value, Path)
            else value,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, default=str))


__all__ = ["identity_features", "main", "selection_score"]


if __name__ == "__main__":
    main()
