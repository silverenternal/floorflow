#!/usr/bin/env python3
"""Train adjacency-aware room classification on cached wall faces."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch

from floorflow.heads.face_semantics import (
    FaceTargets,
    PooledFaceSemanticHead,
    face_semantic_loss,
    face_semantic_metrics,
)
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature-root",
        type=Path,
        default=REPO_ROOT / "outputs/cache/cubicasa5k_face_features_v1",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--min-purity", type=float, default=0.3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs/train/face_semantics_v1",
    )
    return parser.parse_args()


def _paths(root: Path, split: str) -> list[Path]:
    return sorted((root / split).rglob("*.face_features.npz"))


def _load(path: Path) -> dict:
    with np.load(path) as payload:
        return {
            "nodes": payload["nodes"].astype(np.float32),
            "adjacency": payload["adjacency"].astype(np.float32),
            "classes": payload["classes"].astype(np.int64),
            "purity": payload["purity"].astype(np.float32),
        }


def _batch(
    paths: list[Path],
    center: np.ndarray,
    scale: np.ndarray,
    device: torch.device,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[FaceTargets]]:
    payloads = [_load(path) for path in paths]
    nodes = [
        torch.from_numpy((item["nodes"] - center) / scale).to(device)
        for item in payloads
    ]
    adjacency = [
        torch.from_numpy(item["adjacency"]).to(device)
        for item in payloads
    ]
    targets = [
        FaceTargets(
            masks=[],
            adjacency=item["adjacency"],
            classes=item["classes"],
            purity=item["purity"],
        )
        for item in payloads
    ]
    return nodes, adjacency, targets


def _evaluate(
    head: PooledFaceSemanticHead,
    paths: list[Path],
    center: np.ndarray,
    scale: np.ndarray,
    device: torch.device,
    batch_size: int,
    min_purity: float,
) -> dict[str, float]:
    head.eval()
    correct = total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for start in range(0, len(paths), batch_size):
            nodes, adjacency, targets = _batch(
                paths[start : start + batch_size],
                center,
                scale,
                device,
            )
            predictions = head(nodes, adjacency)
            loss, _ = face_semantic_loss(
                predictions,
                targets,
                min_purity=min_purity,
            )
            loss_sum += float(loss)
            metrics = face_semantic_metrics(
                predictions,
                targets,
                min_purity=min_purity,
            )
            face_count = sum(
                int((target.purity >= min_purity).sum())
                for target in targets
            )
            correct += int(round(metrics["face_semantic_accuracy"] * face_count))
            total += face_count
    return {
        "loss": loss_sum / max(1, (len(paths) + batch_size - 1) // batch_size),
        "accuracy": correct / max(1, total),
        "faces": float(total),
    }


def main() -> None:
    args = parse_args()
    train_paths = _paths(args.feature_root, "train")
    val_paths = _paths(args.feature_root, "val")
    test_paths = _paths(args.feature_root, "test")
    first = _load(train_paths[0])["nodes"]
    input_dim = int(first.shape[1])
    train_rows = []
    train_classes = []
    for path in train_paths:
        item = _load(path)
        train_rows.append(item["nodes"])
        train_classes.append(
            item["classes"][item["purity"] >= args.min_purity]
        )
    stacked = np.concatenate(train_rows, axis=0)
    center = stacked.mean(axis=0)
    scale = stacked.std(axis=0)
    scale = np.where(scale > 1.0e-6, scale, 1.0)
    class_counts = np.bincount(
        np.concatenate(train_classes).astype(np.int64),
        minlength=2,
    )
    num_classes = len(class_counts)
    weights = (class_counts.sum() / (len(class_counts) * np.maximum(1, class_counts))).astype(np.float32)
    class_weights = torch.from_numpy(weights).to(args.device)
    head = PooledFaceSemanticHead(
        input_dim,
        len(class_weights),
        hidden_dim=args.hidden,
    ).to(args.device)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_accuracy = -1.0
    history = []
    rng = np.random.default_rng(17)
    for epoch in range(args.epochs):
        head.train()
        order = rng.permutation(len(train_paths))
        losses = []
        for start in range(0, len(order), args.batch_size):
            selected = [train_paths[index] for index in order[start : start + args.batch_size]]
            nodes, adjacency, targets = _batch(
                selected,
                center,
                scale,
                torch.device(args.device),
            )
            predictions = head(nodes, adjacency)
            loss, _ = face_semantic_loss(
                predictions,
                targets,
                min_purity=args.min_purity,
                class_weights=class_weights,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        validation = _evaluate(
            head,
            val_paths,
            center,
            scale,
            torch.device(args.device),
            args.batch_size,
            args.min_purity,
        )
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            **{f"val_{key}": value for key, value in validation.items()},
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if validation["accuracy"] > best_accuracy:
            best_accuracy = validation["accuracy"]
            torch.save(
                {
                    "head": head.state_dict(),
                    "input_dim": input_dim,
                    "num_classes": len(class_weights),
                    "hidden_dim": args.hidden,
                    "feature_center": center,
                    "feature_scale": scale,
                    "min_purity": args.min_purity,
                    "epoch": epoch,
                },
                args.output_dir / "best.pt",
            )
    checkpoint = torch.load(
        args.output_dir / "best.pt",
        map_location=args.device,
        weights_only=False,
    )
    head.load_state_dict(checkpoint["head"])
    test = _evaluate(
        head,
        test_paths,
        center,
        scale,
        torch.device(args.device),
        args.batch_size,
        args.min_purity,
    )
    report = {
        "claim_role": "gt_wall_graph_face_semantics",
        "train_cases": len(train_paths),
        "val_cases": len(val_paths),
        "test_cases": len(test_paths),
        "best_val_accuracy": best_accuracy,
        "test": test,
        "history": history,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    if test["accuracy"] < 0.90:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
