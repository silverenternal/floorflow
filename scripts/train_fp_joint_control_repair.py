#!/usr/bin/env python3
"""Train mask-aware joint repair for missing room and structure controls."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from floorflow.data.control_fusion import nearest_fill_batch
from floorflow.data.control_perturbation import random_spatial_block_mask
from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.heads.coarse_structure_refiner import coarse_structure_control
from floorflow.heads.joint_control_repair import JointControlRepair
from floorflow.objectives.control_repair import (
    masked_focal_dice,
    masked_room_loss,
    room_index_target,
    structure_losses,
)
from floorflow.objectives.differentiable_fevp import differentiable_fevp_terms
from floorflow.representation.coarse_layout import coarse_room_layout_logits
from floorflow.training.distribution_runtime import set_training_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path(".data/cubicasa5k"))
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("outputs/cache/cubicasa5k_full_v2"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--room-size", type=int, default=224)
    parser.add_argument("--structure-size", type=int, default=128)
    parser.add_argument("--room-base-channels", type=int, default=24)
    parser.add_argument("--structure-base-channels", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--max-dropout", type=float, default=0.4)
    parser.add_argument("--room-loss-weight", type=float, default=1.0)
    parser.add_argument("--structure-loss-weight", type=float, default=1.0)
    parser.add_argument("--boundary-loss-weight", type=float, default=0.1)
    parser.add_argument("--differentiable-fevp-weight", type=float, default=0.0)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=191)
    parser.add_argument("--initialize-from", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=19)
    return parser.parse_args()


def _binary_f1(probability: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> float:
    predicted = (probability >= 0.5) & valid
    expected = (target >= 0.5) & valid
    intersection = (predicted & expected).sum()
    denominator = predicted.sum() + expected.sum()
    if int(denominator) == 0:
        return 1.0
    return float(2.0 * intersection / denominator)


def _support_iou(predicted: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> float:
    predicted = predicted & valid
    expected = target & valid
    union = (predicted | expected).sum()
    if int(union) == 0:
        return 1.0
    return float((predicted & expected).sum() / union)


@torch.inference_mode()
def validate(
    model: JointControlRepair,
    loader: DataLoader,
    taxonomy,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.eval()
    device = args.device
    room_ids = torch.tensor(taxonomy.room_ids, device=device)
    results: dict[str, list[float]] = {}
    for fraction in (0.0, 0.1, 0.25):
        room_scores = []
        room_class_scores = []
        wall_scores = []
        opening_scores = []
        for target, footprint, _floor_ids, _names in loader:
            target = target.to(device)
            footprint = footprint.to(device)
            clean_room = coarse_room_layout_logits(
                target,
                taxonomy,
                size=args.room_size,
            )
            clean_structure = coarse_structure_control(
                target,
                footprint,
                taxonomy,
                size=args.structure_size,
            )
            room_missing = random_spatial_block_mask(
                target.shape[0],
                clean_room.shape[-2:],
                fraction,
                generator=torch.Generator().manual_seed(
                    args.seed + 700 + round(fraction * 100)
                ),
            ).to(device)
            structure_missing = random_spatial_block_mask(
                target.shape[0],
                clean_structure.shape[-2:],
                fraction,
                generator=torch.Generator().manual_seed(
                    args.seed + 900 + round(fraction * 100)
                ),
            ).to(device)
            room_input = nearest_fill_batch(clean_room, ~room_missing)
            structure_input = nearest_fill_batch(clean_structure, ~structure_missing)
            output = model(
                room_input,
                structure_input,
                footprint,
                ~room_missing,
                ~structure_missing,
            )
            room_valid_full = F.interpolate(
                room_missing[:, None].float(),
                size=target.shape[-2:],
                mode="nearest",
            ).bool()
            predicted_support = output["room_logits_full"].amax(dim=1) > 0.0
            predicted_labels = room_ids[
                output["room_logits_full"].argmax(dim=1)
            ]
            expected_support = torch.isin(target, room_ids)
            for index in range(target.shape[0]):
                room_scores.append(
                    _support_iou(
                        predicted_support[index],
                        expected_support[index],
                        room_valid_full[index, 0],
                    )
                )
                target_room = torch.isin(target[index], room_ids)
                pixel_valid = target_room & room_valid_full[index, 0]
                if pixel_valid.any():
                    room_class_scores.append(
                        float(
                            (
                                (
                                    predicted_labels[index]
                                    == target[index]
                                )
                                & pixel_valid
                            ).sum()
                            / pixel_valid.sum()
                        )
                    )
            missing_structure = structure_missing[:, None].bool()
            room_count = len(taxonomy.room_ids)
            wall_scores.extend(
                _binary_f1(
                    output["structure"]["wall_logits"][index, 0].sigmoid(),
                    clean_structure[index, room_count],
                    missing_structure[index, 0],
                )
                for index in range(target.shape[0])
            )
            opening_scores.extend(
                _binary_f1(
                    output["structure"]["opening_logits"][index, 0].sigmoid(),
                    clean_structure[index, room_count + 1],
                    missing_structure[index, 0],
                )
                for index in range(target.shape[0])
            )
        key = f"p{round(fraction * 100):03d}"
        results[f"room_support_iou_{key}"] = float(
            sum(room_scores) / max(len(room_scores), 1)
        )
        results[f"room_class_accuracy_{key}"] = float(
            sum(room_class_scores) / max(len(room_class_scores), 1)
        )
        results[f"wall_f1_{key}"] = float(
            sum(wall_scores) / max(len(wall_scores), 1)
        )
        results[f"opening_f1_{key}"] = float(
            sum(opening_scores) / max(len(opening_scores), 1)
        )
    primary = [
        results[f"{metric}_p010"]
        for metric in (
            "room_class_accuracy",
            "wall_f1",
            "opening_f1",
        )
    ] + [
        results[f"{metric}_p025"]
        for metric in (
            "room_class_accuracy",
            "wall_f1",
            "opening_f1",
        )
    ]
    results["selection_score"] = sum(primary) / len(primary)
    return results


def main() -> None:
    args = parse_args()
    set_training_seed(args.seed)
    args.device = torch.device(args.device) if isinstance(args.device, str) else args.device
    train = CubiCasaFloorplanDataset(
        args.data_root,
        "train",
        image_size=256,
        cache_root=args.cache_root,
        limit=args.limit_train,
        augment=False,
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
        "pin_memory": args.device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(train, shuffle=True, **common)
    val_loader = DataLoader(val, shuffle=False, **common)
    room_ids = torch.tensor(taxonomy.room_ids, device=args.device)
    opening_ids = torch.tensor(taxonomy.opening_ids, device=args.device)
    room_count = len(taxonomy.room_ids)
    model = JointControlRepair(
        room_count,
        len(taxonomy.opening_ids),
        room_base_channels=args.room_base_channels,
        structure_base_channels=args.structure_base_channels,
    ).to(args.device)
    if args.initialize_from is not None:
        payload = torch.load(
            args.initialize_from,
            map_location=args.device,
            weights_only=False,
        )
        model.load_state_dict(payload["model"])
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
    history = []
    for epoch in range(args.epochs):
        started = time.perf_counter()
        model.train()
        totals: dict[str, float] = {}
        seen = 0
        for target, footprint, _floor_ids, _names in train_loader:
            target = target.to(args.device)
            footprint = footprint.to(args.device)
            clean_room = coarse_room_layout_logits(
                target,
                taxonomy,
                size=args.room_size,
            )
            clean_structure = coarse_structure_control(
                target,
                footprint,
                taxonomy,
                size=args.structure_size,
            )
            fractions = args.max_dropout * torch.rand(target.shape[0])
            room_missing = random_spatial_block_mask(
                target.shape[0],
                clean_room.shape[-2:],
                fractions,
            ).to(args.device)
            structure_missing = random_spatial_block_mask(
                target.shape[0],
                clean_structure.shape[-2:],
                fractions,
            ).to(args.device)
            room_input = nearest_fill_batch(clean_room, ~room_missing)
            structure_input = nearest_fill_batch(clean_structure, ~structure_missing)
            output = model(
                room_input,
                structure_input,
                footprint,
                ~room_missing,
                ~structure_missing,
            )
            room_missing_full = F.interpolate(
                room_missing[:, None].float(),
                size=target.shape[-2:],
                mode="nearest",
            ).bool()
            room_target = room_index_target(target, room_ids)
            room_loss, room_terms = masked_room_loss(
                output["room_logits_full"],
                room_target,
                room_missing_full,
            )
            structure_loss, structure_terms = structure_losses(
                output["structure"],
                clean_structure,
                structure_missing,
                room_count,
                opening_ids,
            )
            boundary_target = clean_structure[:, room_count : room_count + 2]
            boundary_valid = (
                structure_missing[:, None].bool()
                & (
                    F.max_pool2d(
                        clean_structure[:, room_count : room_count + 2],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                    - clean_structure[:, room_count : room_count + 2]
                ).abs().bool()
            )
            boundary_loss = masked_focal_dice(
                torch.cat(
                    [
                        output["structure"]["wall_logits"],
                        output["structure"]["opening_logits"],
                    ],
                    dim=1,
                ),
                boundary_target,
                boundary_valid,
            )
            soft_fevp = output["room_logits_full"].sum() * 0.0
            soft_fevp_terms = {}
            if args.differentiable_fevp_weight > 0.0:
                soft_fevp, soft_fevp_terms = differentiable_fevp_terms(
                    output["room_logits_full"].softmax(dim=1),
                    F.interpolate(
                        room_input.softmax(dim=1),
                        size=target.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    ),
                    output["structure"]["wall_logits"].sigmoid()[:, 0],
                    output["structure"]["opening_logits"].sigmoid()[:, 0],
                    footprint.float(),
                    room_missing_full.float(),
                )
            loss = (
                args.room_loss_weight * room_loss
                + args.structure_loss_weight * structure_loss
                + args.boundary_loss_weight * boundary_loss
                + args.differentiable_fevp_weight * soft_fevp
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            structure_difference = (
                output["structure_control"] - clean_structure
            ).abs()
            valid_control_mask = (~structure_missing)[:, None]
            valid_consistency = (
                (structure_difference * valid_control_mask).amax().detach()
                if valid_control_mask.any()
                else loss.detach() * 0.0
            )
            terms = {
                "loss": loss.detach(),
                "boundary": boundary_loss.detach(),
                "soft_fevp": soft_fevp.detach(),
                **soft_fevp_terms,
                **room_terms,
                **structure_terms,
                "valid_consistency": valid_consistency,
            }
            for name, value in terms.items():
                totals[name] = totals.get(name, 0.0) + float(value) * target.shape[0]
            seen += target.shape[0]
        scheduler.step()
        metrics = validate(model, val_loader, taxonomy, args)
        record = {
            "epoch": epoch,
            **{f"train_{key}": value / max(1, seen) for key, value in totals.items()},
            **metrics,
            "seconds": time.perf_counter() - started,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        (args.output_dir / "history.json").write_text(
            json.dumps(history, indent=2) + "\n",
            encoding="utf-8",
        )
        checkpoint = {
            "model": model.state_dict(),
            "taxonomy": taxonomy.to_json(),
            "config": vars(args) | {"device": str(args.device)},
            "metrics": metrics,
        }
        torch.save(checkpoint, args.output_dir / "last.pt")
        if metrics["selection_score"] > best:
            best = metrics["selection_score"]
            torch.save(checkpoint, args.output_dir / "best.pt")


if __name__ == "__main__":
    main()
