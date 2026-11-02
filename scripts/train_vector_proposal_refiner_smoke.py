#!/usr/bin/env python3
"""Train dense-to-vector residual refinement from readout proposals."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from floorflow.data.vector_geometry_dataset import VectorGeometryDataset
from floorflow.evaluation.vector_edit_protocol import extract_vector_plan
from floorflow.heads.vector_proposal_refiner import VectorProposalRefiner
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
        default=REPO_ROOT / "outputs/train/vector_proposal_refiner_smoke_v1",
    )
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit-train", type=int, default=128)
    parser.add_argument("--limit-val", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--init-refiner", type=Path)
    parser.add_argument("--start-epoch", type=int, default=0)
    return parser.parse_args()


def _resample(points: np.ndarray, count: int) -> np.ndarray:
    values = np.asarray(points, dtype=np.float32)
    if len(values) < 3:
        return np.zeros((count, 2), dtype=np.float32)
    closed = np.concatenate([values, values[:1]], axis=0)
    lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    samples = np.linspace(0.0, cumulative[-1], count, endpoint=False)
    result = np.empty((count, 2), dtype=np.float32)
    result[:, 0] = np.interp(samples, cumulative, closed[:, 0])
    result[:, 1] = np.interp(samples, cumulative, closed[:, 1])
    return result


def _match(proposal_centers: np.ndarray, target_centers: np.ndarray) -> tuple:
    if len(proposal_centers) == 0 or len(target_centers) == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    cost = np.linalg.norm(
        proposal_centers[:, None] - target_centers[None],
        axis=-1,
    )
    return linear_sum_assignment(cost)


def _proposals(
    batch: dict,
    taxonomy,
    maximums: tuple[int, int, int],
    polygon_points: int,
) -> dict[str, torch.Tensor]:
    max_rooms, max_walls, max_openings = maximums
    room_polygons = torch.zeros(
        len(batch["target"]),
        max_rooms,
        polygon_points,
        2,
    )
    room_valid = torch.zeros(
        len(batch["target"]),
        max_rooms,
        dtype=torch.bool,
    )
    wall_endpoints = torch.zeros(
        len(batch["target"]),
        max_walls,
        2,
        2,
    )
    wall_valid = torch.zeros(
        len(batch["target"]),
        max_walls,
        dtype=torch.bool,
    )
    opening_centers = torch.zeros(
        len(batch["target"]),
        max_openings,
        2,
    )
    opening_valid = torch.zeros(
        len(batch["target"]),
        max_openings,
        dtype=torch.bool,
    )
    scale = float(batch["target"].shape[-1])
    for index, labels in enumerate(batch["target"].cpu().numpy()):
        plan = extract_vector_plan(labels, labels, taxonomy)
        room_centers = np.asarray(
            [
                [room.polygon.centroid.x / scale, room.polygon.centroid.y / scale]
                for room in plan.rooms
            ],
            dtype=np.float32,
        ).reshape(-1, 2)
        room_target_centers = (
            batch["room_polygons"][index].mean(dim=1).numpy()
        )
        room_count = int(batch["room_exists"][index].sum())
        rows, columns = _match(
            room_centers,
            room_target_centers[:room_count],
        )
        for proposal, target in zip(rows, columns, strict=True):
            room_polygons[index, target] = torch.from_numpy(
                _resample(
                    np.asarray(plan.rooms[proposal].polygon.exterior.coords) / scale,
                    polygon_points,
                )
            )
            room_valid[index, target] = True
        wall_centers = np.asarray(
            [
                [
                    (wall.start[0] + wall.end[0]) / (2.0 * scale),
                    (wall.start[1] + wall.end[1]) / (2.0 * scale),
                ]
                for wall in plan.walls
            ],
            dtype=np.float32,
        ).reshape(-1, 2)
        wall_count = int(batch["wall_exists"][index].sum())
        wall_target_centers = (
            batch["wall_endpoints"][index].mean(dim=1).numpy()
        )
        rows, columns = _match(
            wall_centers,
            wall_target_centers[:wall_count],
        )
        for proposal, target in zip(rows, columns, strict=True):
            wall = plan.walls[proposal]
            wall_endpoints[index, target] = torch.tensor(
                [
                    [wall.start[0] / scale, wall.start[1] / scale],
                    [wall.end[0] / scale, wall.end[1] / scale],
                ],
                dtype=torch.float32,
            )
            wall_valid[index, target] = True
        opening_centers_np = np.asarray(
            [
                [opening.center[0] / scale, opening.center[1] / scale]
                for opening in plan.openings
            ],
            dtype=np.float32,
        ).reshape(-1, 2)
        opening_count = int(batch["opening_exists"][index].sum())
        rows, columns = _match(
            opening_centers_np,
            batch["opening_centers"][index, :opening_count].numpy(),
        )
        for proposal, target in zip(rows, columns, strict=True):
            opening = plan.openings[proposal]
            opening_centers[index, target] = torch.tensor(
                [opening.center[0] / scale, opening.center[1] / scale],
                dtype=torch.float32,
            )
            opening_valid[index, target] = True
    return {
        "room_polygons": room_polygons,
        "room_valid": room_valid,
        "wall_endpoints": wall_endpoints,
        "wall_valid": wall_valid,
        "opening_centers": opening_centers,
        "opening_valid": opening_valid,
    }


def _features(backbone, target, footprint, floor_ids):
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


def _masked_mae(predicted, target, valid):
    if not valid.any():
        return predicted.sum() * 0.0
    values = (predicted - target).abs()
    mask = valid
    while mask.ndim < values.ndim:
        mask = mask[..., None]
    return (values * mask).sum() / (mask.expand_as(values).sum().clamp_min(1))


def _run_epoch(refiner, backbone, loader, device, taxonomy, optimizer=None):
    training = optimizer is not None
    refiner.train(training)
    totals = {
        "loss": 0.0,
        "room_baseline_mae": 0.0,
        "room_refined_mae": 0.0,
        "wall_baseline_mae": 0.0,
        "wall_refined_mae": 0.0,
        "opening_baseline_mae": 0.0,
        "opening_refined_mae": 0.0,
        "room_coverage": 0.0,
        "wall_coverage": 0.0,
        "opening_coverage": 0.0,
    }
    seen = 0
    for batch in loader:
        proposals = _proposals(
            batch,
            taxonomy,
            (32, 64, 48),
            16,
        )
        moved = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        proposals = {
            key: value.to(device)
            for key, value in proposals.items()
        }
        features = _features(
            backbone,
            moved["target"],
            moved["footprint"],
            moved["floor_id"],
        )
        outputs = refiner(
            features,
            proposals["room_polygons"],
            proposals["room_valid"],
            proposals["wall_endpoints"],
            proposals["wall_valid"],
            proposals["opening_centers"],
            proposals["opening_valid"],
        )
        room_baseline = _masked_mae(
            proposals["room_polygons"],
            moved["room_polygons"],
            proposals["room_valid"],
        )
        room_refined = _masked_mae(
            outputs["room_polygons"],
            moved["room_polygons"],
            proposals["room_valid"],
        )
        wall_baseline = _masked_mae(
            proposals["wall_endpoints"],
            moved["wall_endpoints"],
            proposals["wall_valid"],
        )
        wall_refined = _masked_mae(
            outputs["wall_endpoints"],
            moved["wall_endpoints"],
            proposals["wall_valid"],
        )
        opening_baseline = _masked_mae(
            proposals["opening_centers"],
            moved["opening_centers"],
            proposals["opening_valid"],
        )
        opening_refined = _masked_mae(
            outputs["opening_centers"],
            moved["opening_centers"],
            proposals["opening_valid"],
        )
        loss = room_refined + wall_refined + opening_refined
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(refiner.parameters(), 1.0)
            optimizer.step()
        batch_size = moved["target"].shape[0]
        seen += batch_size
        for key, value in {
            "loss": loss,
            "room_baseline_mae": room_baseline,
            "room_refined_mae": room_refined,
            "wall_baseline_mae": wall_baseline,
            "wall_refined_mae": wall_refined,
            "opening_baseline_mae": opening_baseline,
            "opening_refined_mae": opening_refined,
        }.items():
            totals[key] += float(value.detach().cpu()) * batch_size
        totals["room_coverage"] += float(proposals["room_valid"].sum())
        totals["wall_coverage"] += float(proposals["wall_valid"].sum())
        totals["opening_coverage"] += float(
            proposals["opening_valid"].sum()
        )
    return {key: value / max(1, seen) for key, value in totals.items()}


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
    refiner = VectorProposalRefiner(
        in_channels=backbone.output.in_channels,
        hidden_dim=args.hidden_dim,
    ).to(device)
    if args.init_refiner is not None:
        initial = torch.load(
            args.init_refiner,
            map_location="cpu",
            weights_only=False,
        )
        refiner.load_state_dict(initial["refiner"])
    optimizer = torch.optim.AdamW(
        refiner.parameters(),
        lr=args.lr,
        weight_decay=1.0e-4,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best = float("inf")
    initial_val = None
    if args.init_refiner is not None:
        initial_val = _run_epoch(
            refiner,
            backbone,
            val_loader,
            device,
            taxonomy,
        )
        best = initial_val["loss"]
        initial_checkpoint = {
            "refiner": refiner.state_dict(),
            "taxonomy": taxonomy.to_json(),
            "backbone_checkpoint": str(args.checkpoint),
            "initial_refiner": str(args.init_refiner),
            "config": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
        }
        torch.save(initial_checkpoint, args.output_dir / "best.pt")
        torch.save(initial_checkpoint, args.output_dir / "last.pt")
    for offset in range(args.epochs):
        epoch = args.start_epoch + offset
        train_metrics = _run_epoch(
            refiner,
            backbone,
            train_loader,
            device,
            taxonomy,
            optimizer,
        )
        val_metrics = _run_epoch(
            refiner,
            backbone,
            val_loader,
            device,
            taxonomy,
        )
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(record)
        print(json.dumps(record), flush=True)
        checkpoint = {
            "refiner": refiner.state_dict(),
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
        (args.output_dir / "progress.json").write_text(
            json.dumps(
                {
                    "status": "running",
                    "initial_val": initial_val,
                    "history": history,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    final = history[-1]["val"]
    checks = {
        "room_mae_improved": (
            final["room_refined_mae"] < final["room_baseline_mae"]
        ),
        "wall_mae_improved": (
            final["wall_refined_mae"] < final["wall_baseline_mae"]
        ),
        "opening_mae_improved": (
            final["opening_refined_mae"] < final["opening_baseline_mae"]
        ),
        "loss_finite": all(
            np.isfinite(row[split]["loss"])
            for row in history
            for split in ("train", "val")
        ),
    }
    report = {
        "status": "complete" if all(checks.values()) else "needs_iteration",
        "train_cases": len(train),
        "val_cases": len(val),
        "initial_refiner": (
            str(args.init_refiner) if args.init_refiner is not None else None
        ),
        "start_epoch": args.start_epoch,
        "initial_val": initial_val,
        "best_val_loss": best,
        "history": history,
        "checks": checks,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": report["status"], "checks": checks}, indent=2))
    raise SystemExit(0 if report["status"] == "complete" else 1)


__all__ = ["main"]


if __name__ == "__main__":
    main()
