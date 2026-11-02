#!/usr/bin/env python3
"""Train dense-query support arbitration with frozen room heads."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from floorflow.data.dataset import CubiCasaRasterControlDataset
from floorflow.evaluation.room_metric_suite import room_metric_suite
from floorflow.heads.dense_query_support_gate import (
    DenseQuerySupportGate,
    SelectiveRoomSupportDeletion,
    render_support_conditioned_room_labels,
    selective_deletion_loss,
    support_gate_loss,
)
from floorflow.heads.predicted_room_identity import room_identity_loss
from floorflow.heads.raster_control_bank import (
    load_raster_instance_control_model,
    load_raster_room_model,
)

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / ".data/cubicasa5k")
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=ROOT / "outputs/cache/cubicasa5k_full_v2_384",
    )
    parser.add_argument(
        "--dense-checkpoint",
        type=Path,
        default=ROOT / "outputs/train/native_room_384_full_32ep_v1/best.pt",
    )
    parser.add_argument(
        "--query-checkpoint",
        type=Path,
        default=ROOT / "outputs/train/room_query_fullres_softroute_screen256_v1/best.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/train/dense_query_support_gate_screen256_v1",
    )
    parser.add_argument("--limit-train", type=int, default=256)
    parser.add_argument("--limit-val", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--fixed-blend", type=float, default=0.75)
    parser.add_argument("--hidden-channels", type=int, default=16)
    parser.add_argument(
        "--mode",
        choices=("class_blend", "selective_deletion"),
        default="class_blend",
    )
    parser.add_argument("--preserve-confidence", type=float, default=0.8)
    parser.add_argument("--keep-threshold", type=float, default=0.5)
    parser.add_argument("--topology-safe", action="store_true")
    parser.add_argument("--min-deletable-component-area", type=int, default=32)
    parser.add_argument("--false-deletion-weight", type=float, default=5.0)
    parser.add_argument("--deletion-weight", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--arbitration-weight", type=float, default=0.5)
    parser.add_argument("--support-weight", type=float, default=0.1)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2905)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def selection_score(metrics: dict[str, float]) -> float:
    return (
        metrics["room_iou"]
        + 0.75 * metrics["instance_f1"]
        + 0.1 * metrics["type_accuracy"]
        + 0.1 * metrics["room_support_iou"]
        - 0.02 * metrics["count_mae"]
    )


@torch.no_grad()
def frozen_probabilities(dense_model, query_model, footprint, image):
    dense_probability = dense_model(
        footprint,
        control_image=image,
    ).softmax(dim=1)
    query_output = query_model(footprint, control_image=image)
    query_probability = query_output["fused_logits"].softmax(dim=1)
    query_support = query_output["mask_logits"].sigmoid().max(dim=1).values
    return dense_probability, query_probability, query_support


def metric_row(
    labels: np.ndarray,
    target: np.ndarray,
    footprint: np.ndarray,
    taxonomy,
) -> dict[str, float]:
    room_ids = np.asarray(taxonomy.room_ids, dtype=np.int64)
    suite = room_metric_suite(
        labels,
        target,
        footprint,
        taxonomy,
        predicted_room_support=np.isin(labels, room_ids),
        target_room_support=np.isin(target, room_ids),
    )
    instance = suite["matched_instance"]
    return {
        "room_iou": float(suite["visible_semantic"]["room_macro_iou"]),
        "instance_f1": float(instance["instance_f1_at_threshold"]),
        "type_accuracy": float(instance["matched_type_accuracy"]),
        "count_mae": float(instance["count_abs_error"]),
        "room_support_iou": float(
            suite["occlusion_aware_support"]["room_support_iou"]
        ),
    }


@torch.inference_mode()
def evaluate(
    gate,
    dense_model,
    query_model,
    loader,
    taxonomy,
    device,
    *,
    mode: str,
    preserve_confidence: float,
    keep_threshold: float,
    topology_safe: bool,
    min_deletable_component_area: int,
) -> dict[str, dict[str, float]]:
    rows = {
        name: []
        for name in (
            "dense",
            "query",
            "blend25",
            "blend50",
            "blend75",
            "blend90",
            "learned_gate",
            "selective_deletion",
        )
    }
    room_ids = torch.tensor(taxonomy.room_ids, device=device)
    gate.eval()
    for target, footprint, image, _floor_ids, _names in loader:
        target = target.to(device)
        footprint = footprint.to(device)
        image = image.to(device)
        dense_probability, query_probability, query_support = frozen_probabilities(
            dense_model,
            query_model,
            footprint,
            image,
        )
        gate_kwargs = {}
        if mode == "selective_deletion":
            gate_kwargs["keep_threshold"] = keep_threshold
        outputs = gate(
            dense_probability,
            query_probability,
            query_support,
            footprint,
            **gate_kwargs,
        )
        probabilities = {
            "dense": dense_probability,
            "query": query_probability,
            "blend25": 0.75 * dense_probability + 0.25 * query_probability,
            "blend50": 0.50 * dense_probability + 0.50 * query_probability,
            "blend75": 0.25 * dense_probability + 0.75 * query_probability,
            "blend90": 0.10 * dense_probability + 0.90 * query_probability,
        }
        if mode == "class_blend":
            probabilities["learned_gate"] = outputs["probability"]
        footprint_4d = footprint[:, None] if footprint.ndim == 3 else footprint
        for name, probability in probabilities.items():
            labels = room_ids[probability.argmax(dim=1)] * footprint_4d[:, 0].long()
            for index in range(target.shape[0]):
                rows[name].append(
                    metric_row(
                        labels[index].cpu().numpy(),
                        target[index].cpu().numpy(),
                        footprint_4d[index, 0].cpu().numpy(),
                        taxonomy,
                    )
                )
        if mode == "selective_deletion":
            labels = render_support_conditioned_room_labels(
                outputs["probability"],
                outputs["keep_mask"],
                room_ids,
                footprint_4d,
                topology_safe=topology_safe,
                minimum_deletable_component_area=(
                    min_deletable_component_area
                ),
            )
            for index in range(target.shape[0]):
                rows["selective_deletion"].append(
                    metric_row(
                        labels[index].cpu().numpy(),
                        target[index].cpu().numpy(),
                        footprint_4d[index, 0].cpu().numpy(),
                        taxonomy,
                    )
                )
        rows = {
            name: values
            for name, values in rows.items()
            if values
        }
    return {
        name: {
            key: float(np.mean([row[key] for row in values]))
            for key in values[0]
        }
        for name, values in rows.items()
    }


def main() -> None:
    args = parse_args()
    if args.device != "cuda:0":
        raise ValueError("dense-query gate training is restricted to cuda:0")
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    train_dataset = CubiCasaRasterControlDataset(
        args.data_root,
        "train",
        image_size=args.image_size,
        cache_root=args.cache_root,
        augment=False,
        limit=args.limit_train,
        image_mode="native",
    )
    taxonomy = train_dataset.taxonomy
    val_dataset = CubiCasaRasterControlDataset(
        args.data_root,
        "val",
        image_size=args.image_size,
        cache_root=args.cache_root,
        augment=False,
        limit=args.limit_val,
        image_mode="native",
    )
    dense_model = load_raster_room_model(
        args.dense_checkpoint,
        taxonomy,
        device,
    )
    query_model = load_raster_instance_control_model(
        args.query_checkpoint,
        taxonomy,
        device,
    )
    dense_model.requires_grad_(False)
    query_model.requires_grad_(False)
    gate_class = (
        SelectiveRoomSupportDeletion
        if args.mode == "selective_deletion"
        else DenseQuerySupportGate
    )
    gate_kwargs = {
        "fixed_blend": args.fixed_blend,
        "hidden_channels": args.hidden_channels,
    }
    if args.mode == "selective_deletion":
        gate_kwargs["preserve_confidence"] = args.preserve_confidence
    gate = gate_class(
        len(taxonomy.room_ids),
        **gate_kwargs,
    ).to(device)
    optimizer = torch.optim.AdamW(
        gate.parameters(),
        lr=args.learning_rate,
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
        gate.train()
        running = 0.0
        component_sums: dict[str, float] = {}
        seen = 0
        for target, footprint, image, _floor_ids, _names in train_loader:
            target = target.to(device)
            footprint = footprint.to(device)
            image = image.to(device)
            dense_probability, query_probability, query_support = frozen_probabilities(
                dense_model,
                query_model,
                footprint,
                image,
            )
            gate_kwargs = (
                {"keep_threshold": args.keep_threshold}
                if args.mode == "selective_deletion"
                else {}
            )
            outputs = gate(
                dense_probability,
                query_probability,
                query_support,
                footprint,
                **gate_kwargs,
            )
            if args.mode == "selective_deletion":
                loss, components = selective_deletion_loss(
                    outputs,
                    target,
                    footprint,
                    taxonomy.room_ids,
                    false_deletion_weight=args.false_deletion_weight,
                    deletion_weight=args.deletion_weight,
                    support_weight=args.support_weight,
                )
            else:
                identity_loss, identity_parts = room_identity_loss(
                    outputs["probability"].clamp_min(1.0e-8).log(),
                    target,
                    footprint,
                    taxonomy.room_ids,
                    dice_weight=0.5,
                    boundary_weight=0.5,
                )
                arbitration_loss, arbitration_parts = support_gate_loss(
                    outputs,
                    target,
                    dense_probability,
                    query_probability,
                    footprint,
                    taxonomy.room_ids,
                    arbitration_weight=args.arbitration_weight,
                    support_weight=args.support_weight,
                )
                loss = identity_loss + arbitration_loss
                components = {**identity_parts, **arbitration_parts}
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gate.parameters(), args.gradient_clip)
            optimizer.step()
            running += float(loss.detach()) * target.shape[0]
            seen += target.shape[0]
            for key, value in components.items():
                component_sums[key] = component_sums.get(key, 0.0) + float(
                    value.detach().cpu()
                )
        validation = evaluate(
            gate,
            dense_model,
            query_model,
            val_loader,
            taxonomy,
            device,
            mode=args.mode,
            preserve_confidence=args.preserve_confidence,
            keep_threshold=args.keep_threshold,
            topology_safe=args.topology_safe,
            min_deletable_component_area=(
                args.min_deletable_component_area
            ),
        )
        arm_name, arm_metrics = max(
            validation.items(),
            key=lambda item: selection_score(item[1]),
        )
        row = {
            "epoch": epoch,
            "train_loss": running / max(1, seen),
            "train_components": {
                key: value / max(1, len(train_loader))
                for key, value in sorted(component_sums.items())
            },
            "best_arm": arm_name,
            "selection_score": selection_score(arm_metrics),
            "validation": validation,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        state = {
            "model": gate.state_dict(),
            "config": vars(args),
            "taxonomy": taxonomy.to_json(),
            "metrics": validation,
        }
        torch.save(state, args.output_dir / "last.pt")
        if row["selection_score"] > best_score:
            best_score = row["selection_score"]
            torch.save(state, args.output_dir / "best.pt")
    report = {
        "status": "complete",
        "scope": "learned dense-query support arbitration",
            "mode": args.mode,
        "condition_mode": "predicted_control",
        "target_used_at_inference": False,
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


if __name__ == "__main__":
    main()
