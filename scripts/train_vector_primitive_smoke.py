#!/usr/bin/env python3
"""Train a vector primitive head on frozen FloorFlow spatial features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from floorflow.data.vector_geometry_dataset import VectorGeometryDataset
from floorflow.heads.vector_primitive_head import VectorPrimitiveHead
from floorflow.training.checkpoint import load_checkpoint_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--geometry-root",
        type=Path,
        default=REPO_ROOT / "outputs/cache/cubicasa5k_geometry_v1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs/train/vector_primitive_smoke_v1",
    )
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit-train", type=int, default=128)
    parser.add_argument("--limit-val", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _features(
    backbone,
    target: torch.Tensor,
    footprint: torch.Tensor,
    floor_ids: torch.Tensor,
) -> torch.Tensor:
    zero = torch.zeros(
        target.shape[0],
        backbone.target_channels,
        *target.shape[-2:],
        device=target.device,
    )
    with torch.no_grad():
        return backbone._encode_features(
            zero,
            torch.zeros(target.shape[0], device=target.device),
            footprint,
            floor_ids=floor_ids,
        )


def _match(
    predicted: torch.Tensor,
    target: torch.Tensor,
    target_exists: torch.Tensor,
    *,
    class_logits: torch.Tensor | None = None,
    target_classes: torch.Tensor | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    pairs = []
    for batch in range(predicted.shape[0]):
        count = int(target_exists[batch].sum().item())
        if count == 0:
            pairs.append(
                (
                    np.zeros(0, dtype=np.int64),
                    np.zeros(0, dtype=np.int64),
                )
            )
            continue
        pred_xy = predicted[batch]
        target_xy = target[batch, :count]
        if pred_xy.ndim == 4 or (
            pred_xy.ndim == 3 and pred_xy.shape[-1] == 2
        ):
            pred_centers = pred_xy.mean(dim=1)
            target_centers = target_xy.mean(dim=1)
        else:
            pred_centers = pred_xy
            target_centers = target_xy
        cost = (
            torch.cdist(pred_centers, target_centers)
            .detach()
            .cpu()
            .numpy()
        )
        if class_logits is not None and target_classes is not None:
            predicted_labels = class_logits[batch].argmax(dim=-1).cpu()
            mismatch = (
                predicted_labels[:, None]
                != target_classes[batch, :count].detach().cpu()[None]
            ).float().numpy()
            cost = cost + mismatch
        rows, columns = linear_sum_assignment(cost)
        pairs.append((rows.astype(np.int64), columns.astype(np.int64)))
    return pairs


def _exists_loss(
    logits: torch.Tensor,
    pairs: list[tuple[np.ndarray, np.ndarray]],
) -> torch.Tensor:
    expected = torch.zeros_like(logits)
    for batch, (rows, _columns) in enumerate(pairs):
        if rows.size:
            expected[batch, torch.from_numpy(rows).to(logits.device)] = 1.0
    positive = expected.sum().clamp_min(1.0)
    negative = (1.0 - expected).sum().clamp_min(1.0)
    return F.binary_cross_entropy_with_logits(
        logits,
        expected,
        pos_weight=(negative / positive).sqrt().clamp(1.0, 10.0),
    )


def _matched_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    pairs: list[tuple[np.ndarray, np.ndarray]],
) -> torch.Tensor:
    losses = []
    for batch, (rows, columns) in enumerate(pairs):
        if not rows.size:
            continue
        rows_t = torch.from_numpy(rows).to(predicted.device)
        columns_t = torch.from_numpy(columns).to(predicted.device)
        losses.append(
            F.smooth_l1_loss(
                predicted[batch, rows_t],
                target[batch, columns_t],
            )
        )
    return torch.stack(losses).mean() if losses else predicted.sum() * 0.0


def _class_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pairs: list[tuple[np.ndarray, np.ndarray]],
) -> torch.Tensor:
    losses = []
    for batch, (rows, columns) in enumerate(pairs):
        if not rows.size:
            continue
        rows_t = torch.from_numpy(rows).to(logits.device)
        columns_t = torch.from_numpy(columns).to(logits.device)
        losses.append(
            F.cross_entropy(
                logits[batch, rows_t],
                targets[batch, columns_t],
            )
        )
    return torch.stack(losses).mean() if losses else logits.sum() * 0.0


def _detection_metrics(
    logits: torch.Tensor,
    pairs: list[tuple[np.ndarray, np.ndarray]],
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    active = logits.sigmoid() > 0.5
    true_positive = 0
    predicted_count = 0
    target_count = 0
    coordinate_error = []
    for batch, (rows, columns) in enumerate(pairs):
        predicted_count += int(active[batch].sum().item())
        target_count += int(rows.size)
        if not rows.size:
            continue
        rows_t = torch.from_numpy(rows).to(predicted.device)
        columns_t = torch.from_numpy(columns).to(predicted.device)
        true_positive += int(active[batch, rows_t].sum().item())
        coordinate_error.append(
            (predicted[batch, rows_t] - target[batch, columns_t])
            .abs()
            .mean()
        )
    precision = true_positive / max(1, predicted_count)
    recall = true_positive / max(1, target_count)
    return {
        "true_positive": float(true_positive),
        "predicted": float(predicted_count),
        "targets": float(target_count),
        "precision": precision,
        "recall": recall,
        "f1": (
            2.0 * precision * recall / max(1.0e-6, precision + recall)
        ),
        "coordinate_mae": (
            float(torch.stack(coordinate_error).mean().detach().cpu())
            if coordinate_error
            else 0.0
        ),
    }


def _objective(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, float]]:
    room_polygons = outputs["room_polygons"]
    wall_endpoints = outputs["wall_endpoints"]
    opening_centers = outputs["opening_centers"]
    room_pairs = _match(
        room_polygons,
        batch["room_polygons"],
        batch["room_exists"],
        class_logits=outputs["room_class"],
        target_classes=batch["room_classes"],
    )
    wall_pairs = _match(
        wall_endpoints,
        batch["wall_endpoints"],
        batch["wall_exists"],
    )
    opening_pairs = _match(
        opening_centers,
        batch["opening_centers"],
        batch["opening_exists"],
        class_logits=outputs["opening_class"],
        target_classes=batch["opening_classes"],
    )
    room_loss = (
        1.0 * _exists_loss(outputs["room_exists"], room_pairs)
        + 5.0 * _matched_loss(room_polygons, batch["room_polygons"], room_pairs)
        + 0.5 * _class_loss(outputs["room_class"], batch["room_classes"], room_pairs)
    )
    wall_loss = (
        1.0 * _exists_loss(outputs["wall_exists"], wall_pairs)
        + 5.0 * _matched_loss(
            wall_endpoints,
            batch["wall_endpoints"],
            wall_pairs,
        )
        + F.smooth_l1_loss(
            outputs["wall_thickness"].sigmoid(),
            batch["wall_thickness"],
        )
        + F.binary_cross_entropy_with_logits(
            outputs["wall_exterior"],
            batch["wall_exterior"],
        )
    )
    opening_loss = (
        1.0 * _exists_loss(outputs["opening_exists"], opening_pairs)
        + 5.0 * _matched_loss(
            opening_centers,
            batch["opening_centers"],
            opening_pairs,
        )
        + F.smooth_l1_loss(
            outputs["opening_widths"].sigmoid(),
            batch["opening_widths"],
        )
        + 0.5
        * _class_loss(
            outputs["opening_class"],
            batch["opening_classes"],
            opening_pairs,
        )
    )
    total = room_loss + wall_loss + opening_loss
    room_detection = _detection_metrics(
        outputs["room_exists"],
        room_pairs,
        room_polygons,
        batch["room_polygons"],
    )
    wall_detection = _detection_metrics(
        outputs["wall_exists"],
        wall_pairs,
        wall_endpoints,
        batch["wall_endpoints"],
    )
    opening_detection = _detection_metrics(
        outputs["opening_exists"],
        opening_pairs,
        opening_centers,
        batch["opening_centers"],
    )
    metrics = {
        **{
            f"room_{key}": value
            for key, value in room_detection.items()
        },
        **{
            f"wall_{key}": value
            for key, value in wall_detection.items()
        },
        **{
            f"opening_{key}": value
            for key, value in opening_detection.items()
        },
    }
    return total, {
        "room_loss": room_loss.detach(),
        "wall_loss": wall_loss.detach(),
        "opening_loss": opening_loss.detach(),
    }, metrics


def _run_epoch(
    head: VectorPrimitiveHead,
    backbone,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    training = optimizer is not None
    head.train(training)
    totals = {
        "loss": 0.0,
        "room_loss": 0.0,
        "wall_loss": 0.0,
        "opening_loss": 0.0,
        **{
            f"{family}_{key}": 0.0
            for family in ("room", "wall", "opening")
            for key in (
                "true_positive",
                "predicted",
                "targets",
                "precision",
                "recall",
                "f1",
                "coordinate_mae",
            )
        },
    }
    seen = 0
    for batch in loader:
        moved = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        features = _features(
            backbone,
            moved["target"],
            moved["footprint"],
            moved["floor_id"],
        )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            outputs = head(features)
        outputs = {key: value.float() for key, value in outputs.items()}
        loss, components, metrics = _objective(outputs, moved)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
        batch_size = moved["target"].shape[0]
        seen += batch_size
        totals["loss"] += float(loss.detach().cpu()) * batch_size
        for key, value in components.items():
            totals[key] += float(value.cpu()) * batch_size
        for key, value in metrics.items():
            totals[key] += value
    result = {key: value / max(1, seen) for key, value in totals.items()}
    return result


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    backbone, _config, taxonomy, _ = load_checkpoint_model(
        args.checkpoint,
        device,
    )
    backbone.requires_grad_(False)
    train = VectorGeometryDataset(
        args.geometry_root,
        "train",
        limit=args.limit_train,
    )
    val = VectorGeometryDataset(
        args.geometry_root,
        "val",
        limit=args.limit_val,
    )
    train_loader = DataLoader(
        train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    head = VectorPrimitiveHead(
        in_channels=backbone.output.in_channels,
        room_classes=len(taxonomy.room_ids),
        opening_classes=max(1, len(taxonomy.opening_ids)),
        hidden_dim=args.hidden_dim,
    ).to(device)
    priors = {}
    sample_count = min(64, len(train))
    for family, maximum in (
        ("room", head.max_rooms),
        ("wall", head.max_walls),
        ("opening", head.max_openings),
    ):
        ratio = float(
            np.mean(
                [
                    float(train[index][f"{family}_exists"].sum())
                    / float(maximum)
                    for index in range(sample_count)
                ]
            )
        )
        ratio = min(0.49, max(0.01, ratio))
        with torch.no_grad():
            getattr(head, f"{family}_exists").bias.fill_(
                float(np.log(ratio / (1.0 - ratio)))
            )
        priors[f"{family}_prior"] = ratio
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=args.lr,
        weight_decay=1.0e-4,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    history = []
    for epoch in range(args.epochs):
        train_metrics = _run_epoch(
            head,
            backbone,
            train_loader,
            device,
            optimizer,
        )
        val_metrics = _run_epoch(
            head,
            backbone,
            val_loader,
            device,
            None,
        )
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        checkpoint = {
            "head": head.state_dict(),
            "taxonomy": taxonomy.to_json(),
            "backbone_checkpoint": str(args.checkpoint),
            "config": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
        }
        torch.save(checkpoint, args.output_dir / "last.pt")
        if val_metrics["loss"] < best:
            best = val_metrics["loss"]
            torch.save(checkpoint, args.output_dir / "best.pt")
    report = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "train_cases": len(train),
        "val_cases": len(val),
        "best_val_loss": best,
        "history": history,
        "checks": {
            "loss_finite": all(
                np.isfinite(row[split]["loss"])
                for row in history
                for split in ("train", "val")
            ),
            "train_loss_decreased": (
                len(history) == 1
                or history[-1]["train"]["loss"] < history[0]["train"]["loss"]
            ),
            "all_primitive_families_observed": all(
                history[0]["train"][f"{family}_targets"] > 0
                for family in ("room", "wall", "opening")
            ),
            "all_primitive_families_predicted": all(
                history[-1]["val"][f"{family}_predicted"] > 0
                for family in ("room", "wall", "opening")
            ),
        },
        "existence_priors": priors,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "complete", "checks": report["checks"]}, indent=2))
    raise SystemExit(0 if all(report["checks"].values()) else 1)


__all__ = ["main"]


if __name__ == "__main__":
    main()
