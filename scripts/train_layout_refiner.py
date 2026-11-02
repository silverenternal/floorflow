"""Train the second-stage coarse-layout room-boundary refiner."""

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
from floorflow.data.retrieval_control_dataset import RetrievalControlDataset
from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.room_metric_suite import room_metric_suite
from floorflow.heads.coarse_layout_refiner import CoarseLayoutRefiner
from floorflow.heads.room_program import (
    gate_room_labels_to_program,
    room_presence_target,
)
from floorflow.objectives.control_repair import room_index_target
from floorflow.objectives.topology_closure import room_transition_loss
from floorflow.representation.coarse_layout import coarse_room_layout_logits
from floorflow.training.distribution_runtime import set_training_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path(".data/cubicasa5k"))
    parser.add_argument("--cache-root", type=Path, default=Path("outputs/cache/cubicasa5k_full_v2"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/train/layout_refiner_v1"))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--layout-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=191)
    parser.add_argument("--control-cache", type=Path, default=None)
    parser.add_argument(
        "--retrieval-controls",
        action="store_true",
        help="Use leave-one-out train contour retrieval instead of target controls.",
    )
    parser.add_argument(
        "--retrieval-metric",
        choices=("cosine", "mask_iou", "hybrid"),
        default="cosine",
    )
    parser.add_argument("--initialize-from", type=Path, default=None)
    parser.add_argument("--boundary-weight", type=float, default=0.0)
    parser.add_argument(
        "--residual-layout",
        action="store_true",
        help="Predict zero-initialized corrections on top of the control layout.",
    )
    return parser.parse_args()


def room_loss(logits: torch.Tensor, labels: torch.Tensor, room_ids: torch.Tensor) -> torch.Tensor:
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
    one_hot = F.one_hot(target.clamp_min(0), len(room_ids)).permute(0, 3, 1, 2).float()
    valid = (target >= 0)[:, None].float()
    one_hot = one_hot * valid
    intersection = (probability * one_hot).sum(dim=(0, 2, 3))
    denominator = (probability * valid).sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
    dice = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    return ce + dice


def unpack_batch(batch):
    if len(batch) == 5:
        return (*batch[:4], batch[4])
    return (*batch, None)


def coarse_control(target, footprint, controls, taxonomy, layout_size, device):
    if controls is not None:
        return controls["room_logits"].to(device)
    return coarse_room_layout_logits(target, taxonomy, size=layout_size)


@torch.no_grad()
def evaluate_model(model, loader, taxonomy, room_ids, args):
    model.eval()
    rows = []
    for batch in loader:
        target, footprint, _floor_ids, _names, controls = unpack_batch(batch)
        target = target.to(args.device)
        footprint = footprint.to(args.device)
        coarse = coarse_control(
            target,
            footprint,
            controls,
            taxonomy,
            args.layout_size,
            args.device,
        )
        logits = model(coarse, footprint)
        prediction = gate_room_labels_to_program(
            room_ids[logits.argmax(dim=1)],
            (
                controls["program"].to(args.device)
                if controls is not None
                else room_presence_target(target, taxonomy)
            ),
            taxonomy.room_ids,
        )
        prediction = prediction * (footprint > 0.5)
        for index in range(target.shape[0]):
            target_room = torch.isin(target[index], room_ids)
            prediction_room = torch.isin(prediction[index], room_ids)
            report = room_metric_suite(
                prediction[index].cpu().numpy(),
                target[index].cpu().numpy(),
                footprint[index].cpu().numpy(),
                taxonomy,
                predicted_room_support=prediction_room.cpu().numpy(),
                predicted_instance_support=prediction_room.cpu().numpy(),
                target_room_support=target_room.cpu().numpy(),
            )
            rows.append(report)
    keys = (
        "room_macro_iou",
        "instance_geometric_iou",
        "instance_room_f1_at_0_5",
        "instance_type_accuracy",
        "support_instance_geometric_iou",
    )
    summary = {}
    metric_map = {
        "instance_geometric_iou": "matched_geometric_iou",
        "instance_room_f1_at_0_5": "instance_f1_at_threshold",
        "instance_type_accuracy": "matched_type_accuracy",
    }
    for key in keys:
        values = []
        for report in rows:
            if key == "room_macro_iou":
                values.append(report["visible_semantic"]["room_macro_iou"])
            elif key == "support_instance_geometric_iou":
                values.append(report["support_instance"]["matched_geometric_iou"])
            else:
                values.append(report["matched_instance"][metric_map[key]])
        summary[key] = float(np.mean(values))
    return summary


def main() -> None:
    args = parse_args()
    set_training_seed(args.seed)
    args.device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    taxonomy = FloorplanTaxonomy.read(args.cache_root / "taxonomy.json")
    train = CubiCasaFloorplanDataset(
        args.data_root,
        "train",
        image_size=args.image_size,
        cache_root=args.cache_root,
        limit=args.limit_train,
        augment=args.control_cache is None,
    )
    val = CubiCasaFloorplanDataset(
        args.data_root,
        "val",
        image_size=args.image_size,
        cache_root=args.cache_root,
        limit=args.limit_val,
        augment=False,
    )
    if args.control_cache is not None:
        train = CachedControlDataset(train, args.control_cache, "train")
        val = CachedControlDataset(val, args.control_cache, "val")
    elif args.retrieval_controls:
        train = RetrievalControlDataset(
            train,
            args.cache_root,
            "train",
            metric=args.retrieval_metric,
            control_size=args.layout_size,
        )
        val = RetrievalControlDataset(
            val,
            args.cache_root,
            "val",
            metric=args.retrieval_metric,
            control_size=args.layout_size,
        )
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    model = CoarseLayoutRefiner(
        len(taxonomy.room_ids),
        residual=args.residual_layout,
    ).to(args.device)
    if args.initialize_from is not None:
        payload = torch.load(
            args.initialize_from,
            map_location=args.device,
            weights_only=False,
        )
        model.load_state_dict(payload["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1.0e-4)
    room_ids = torch.tensor(taxonomy.room_ids, device=args.device)
    best = -1.0
    history = []
    for epoch in range(args.epochs):
        started = time.perf_counter()
        model.train()
        total = 0.0
        total_boundary = 0.0
        seen = 0
        for batch in train_loader:
            target, footprint, _floor_ids, _names, controls = unpack_batch(batch)
            target = target.to(args.device)
            footprint = footprint.to(args.device)
            coarse = coarse_control(
                target,
                footprint,
                controls,
                taxonomy,
                args.layout_size,
                args.device,
            )
            logits = model(coarse, footprint)
            semantic_loss = room_loss(logits, target, room_ids)
            boundary_loss = room_transition_loss(
                logits,
                room_index_target(target, room_ids),
            )
            loss = semantic_loss + args.boundary_weight * boundary_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * target.shape[0]
            total_boundary += float(boundary_loss.detach()) * target.shape[0]
            seen += target.shape[0]
        metrics = evaluate_model(model, val_loader, taxonomy, room_ids, args)
        record = {
            "epoch": epoch,
            "train_loss": total / max(1, seen),
            "train_boundary_loss": total_boundary / max(1, seen),
            "seconds": time.perf_counter() - started,
            **metrics,
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
            "layout_size": args.layout_size,
            "residual": args.residual_layout,
            "seed": args.seed,
            "metrics": metrics,
        }
        torch.save(checkpoint, args.output_dir / "last.pt")
        selection = 0.5 * metrics["room_macro_iou"] + 0.5 * metrics[
            "instance_room_f1_at_0_5"
        ]
        if selection > best:
            best = selection
            torch.save(checkpoint, args.output_dir / "best.pt")


if __name__ == "__main__":
    main()
