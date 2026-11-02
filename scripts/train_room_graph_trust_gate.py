#!/usr/bin/env python3
"""Train a cheap gate that predicts when recursive graph geometry helps."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from floorflow.recursion.graph_adapter import RecursiveSemanticGraphAdapter
from floorflow.recursion.trust_gate import RoomGraphTrustGate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature-cache",
        type=Path,
        default=Path(
            "outputs/train/recursive_graph_adapter_1024_v1/feature_cache.pt"
        ),
    )
    parser.add_argument(
        "--graph-checkpoint",
        type=Path,
        default=Path("outputs/train/recursive_graph_adapter_1024_v1/best.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/train/room_graph_trust_gate_v1"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("outputs/reports/room_graph_trust_gate_v1.json"),
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=3.0e-3)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_graph_adapter(path: Path, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = payload.get("config", {})
    adapter = RecursiveSemanticGraphAdapter(
        int(payload["num_classes"]),
        hidden_size=int(config.get("hidden_size", 64)),
        heads=int(config.get("heads", 4)),
        router_count=int(config.get("router_count", 8)),
        top_k=int(config.get("top_k", 4)),
        blend=float(config.get("blend", 0.15)),
    )
    adapter.load_state_dict(payload["model"])
    return adapter.to(device).eval()


def node_iou(prediction: torch.Tensor, target: torch.Tensor) -> float:
    valid = target >= 0
    if not valid.any():
        return 0.0
    prediction = prediction[valid]
    target = target[valid]
    values = []
    for class_id in target.unique():
        expected = target == class_id
        predicted = prediction == class_id
        values.append(
            float((expected & predicted).sum())
            / max(1.0, float((expected | predicted).sum()))
        )
    return sum(values) / len(values)


def trust_features(base: torch.Tensor, graph: torch.Tensor) -> torch.Tensor:
    base_probability = base.float().softmax(dim=-1)
    graph_probability = graph.float().softmax(dim=-1)
    base_confidence = base_probability.max(dim=-1).values
    graph_confidence = graph_probability.max(dim=-1).values
    base_entropy = -(
        base_probability.clamp_min(1.0e-8)
        * base_probability.clamp_min(1.0e-8).log()
    ).sum(dim=-1) / math.log(max(2, base.shape[-1]))
    graph_entropy = -(
        graph_probability.clamp_min(1.0e-8)
        * graph_probability.clamp_min(1.0e-8).log()
    ).sum(dim=-1) / math.log(max(2, graph.shape[-1]))
    return torch.stack(
        [
            base_confidence.mean(),
            graph_confidence.mean(),
            (graph_confidence - base_confidence).mean(),
            (base.argmax(dim=-1) == graph.argmax(dim=-1)).float().mean(),
            base_entropy.mean(),
            graph_entropy.mean(),
            (base_confidence >= 0.5).float().mean(),
            (graph_confidence >= 0.5).float().mean(),
        ]
    )


@torch.no_grad()
def prepare(cases, adapter, device):
    features = []
    labels = []
    details = []
    for raw_case in cases:
        case = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in raw_case.items()
        }
        correction, _ = adapter.forward_features(
            case["features"],
            case["adjacency"],
            case["topology"],
            recursion_strength=case["strength"],
        )
        base = case["base_scores"]
        graph = base + correction[0]
        prediction = graph.argmax(dim=-1)
        target = case["labels"]
        base_iou = node_iou(base.argmax(dim=-1), target)
        graph_iou = node_iou(prediction, target)
        features.append(trust_features(base, graph))
        labels.append(float(graph_iou > base_iou + 1.0e-6))
        details.append(
            {
                "name": raw_case["name"],
                "base_iou": base_iou,
                "graph_iou": graph_iou,
            }
        )
    return torch.stack(features), torch.tensor(labels, device=device), details


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    cache = torch.load(
        args.feature_cache,
        map_location="cpu",
        weights_only=False,
    )
    adapter = load_graph_adapter(args.graph_checkpoint, device)
    train_x, train_y, train_rows = prepare(cache["train"], adapter, device)
    val_x, val_y, val_rows = prepare(cache["val"], adapter, device)
    gate = RoomGraphTrustGate().to(device)
    positive = train_y.mean().clamp(0.05, 0.95)
    loss_weight = ((1.0 - positive) / positive).detach()
    optimizer = torch.optim.AdamW(gate.parameters(), lr=args.lr)
    best = -1.0
    history = []
    for epoch in range(args.epochs):
        gate.train()
        logits = gate(train_x)
        loss = F.binary_cross_entropy_with_logits(
            logits,
            train_y,
            pos_weight=loss_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        gate.eval()
        with torch.no_grad():
            val_prediction = gate.trust(val_x) >= 0.5
            accuracy = float((val_prediction == val_y.bool()).float().mean())
            precision = float(
                
                    (val_prediction & val_y.bool()).sum()
                    / val_prediction.sum().clamp_min(1)
                
            )
            recall = float(
                
                    (val_prediction & val_y.bool()).sum()
                    / val_y.bool().sum().clamp_min(1)
                
            )
        if accuracy > best:
            best = accuracy
            args.output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": gate.state_dict(),
                    "config": {
                        "feature_size": train_x.shape[1],
                        "hidden_size": 32,
                    },
                },
                args.output_dir / "best.pt",
            )
        if epoch % 50 == 0 or epoch == args.epochs - 1:
            history.append(
                {
                    "epoch": epoch,
                    "loss": float(loss.detach()),
                    "val_accuracy": accuracy,
                    "val_precision": precision,
                    "val_recall": recall,
                    "train_positive_fraction": float(train_y.mean()),
                }
            )
            print(json.dumps(history[-1]), flush=True)
    report = {
        "status": "complete",
        "train_cases": len(train_rows),
        "val_cases": len(val_rows),
        "best_val_accuracy": best,
        "history": history,
        "selection": "train/val only; no target-derived inference input",
        "train": train_rows,
        "val": val_rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
