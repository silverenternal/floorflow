#!/usr/bin/env python3
"""Train the learned direct/recursive/split complexity policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from floorflow.data.dataset import CubiCasaFloorplanDataset, load_or_rasterize
from floorflow.recursion.complexity import (
    POLICY_ACTION_NAMES,
    POLICY_INPUT_FEATURE_NAMES,
    PolicyLabelCalibration,
    RecursionPolicy,
    complexity_features,
    complexity_score,
    derive_policy_labels,
    policy_loss,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path(".data/cubicasa5k"))
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("outputs/cache/cubicasa5k_full_v2"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/train/recursion_policy_v1"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("outputs/reports/recursion_policy_v1.json"),
    )
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=3.0e-3)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def collect(
    dataset: CubiCasaFloorplanDataset,
) -> tuple[torch.Tensor, list[str]]:
    features = []
    names = []
    for sample in dataset.samples:
        target, footprint = load_or_rasterize(
            sample,
            image_size=dataset.image_size,
            taxonomy=dataset.taxonomy,
        )
        features.append(
            complexity_features(
                torch.from_numpy(footprint)[None],
                torch.from_numpy(target)[None],
                dataset.taxonomy,
            )[0]
        )
        names.append(f"{sample.relative_case}/{sample.floor_id}")
    return torch.stack(features), names


def action_metrics(
    action: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, object]:
    confusion = np.zeros((3, 3), dtype=np.int64)
    for expected, predicted in zip(target.tolist(), action.tolist()):
        confusion[int(expected), int(predicted)] += 1
    per_class = {}
    for index, name in enumerate(POLICY_ACTION_NAMES):
        support = int(confusion[index].sum())
        predicted = int(confusion[:, index].sum())
        correct = int(confusion[index, index])
        per_class[name] = {
            "support": support,
            "predicted": predicted,
            "recall": correct / max(1, support),
            "precision": correct / max(1, predicted),
        }
    return {
        "accuracy": float((action == target).float().mean()),
        "confusion": confusion.tolist(),
        "per_action": per_class,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    train_dataset = CubiCasaFloorplanDataset(
        args.data_root,
        "train",
        image_size=256,
        cache_root=args.cache_root,
        limit=args.limit_train,
        augment=False,
    )
    val_dataset = CubiCasaFloorplanDataset(
        args.data_root,
        "val",
        image_size=256,
        cache_root=args.cache_root,
        limit=args.limit_val,
        augment=False,
    )
    train_full_features, _train_names = collect(train_dataset)
    val_full_features, val_names = collect(val_dataset)
    calibration = PolicyLabelCalibration.fit(train_full_features)
    train_labels = derive_policy_labels(train_full_features, calibration)
    val_labels = derive_policy_labels(val_full_features, calibration)
    train_features = train_full_features[:, : len(POLICY_INPUT_FEATURE_NAMES)]
    val_features = val_full_features[:, : len(POLICY_INPUT_FEATURE_NAMES)]
    mean = train_features.mean(dim=0)
    scale = train_features.std(dim=0).clamp_min(1.0e-6)
    policy = RecursionPolicy().to(device)
    policy.set_standardizer(mean, scale)
    counts = torch.bincount(train_labels, minlength=3).float()
    class_weights = (counts.sum() / (3.0 * counts.clamp_min(1.0))).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr)
    best_accuracy = -1.0
    best_state = None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_features = train_features.to(device)
    train_labels = train_labels.to(device)
    for epoch in range(args.epochs):
        policy.train()
        optimizer.zero_grad(set_to_none=True)
        loss = policy_loss(
            policy(train_features),
            train_labels,
            class_weights=class_weights,
        )
        loss.backward()
        optimizer.step()
        decision = policy.decide(val_features.to(device))
        accuracy = float(
            (decision["action"].cpu() == val_labels).float().mean()
        )
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in policy.state_dict().items()
            }
        if epoch % 50 == 0 or epoch == args.epochs - 1:
            print(
                json.dumps(
                    {
                        "epoch": epoch,
                        "loss": float(loss.detach()),
                        "val_accuracy": accuracy,
                    }
                ),
                flush=True,
            )
    if best_state is not None:
        policy.load_state_dict(best_state)
    policy.eval()
    val_scores = complexity_score(val_full_features)
    decision = policy.decide(val_features.to(device))
    action = decision["action"].cpu()
    expected_strength = val_labels.float() / 2.0
    strength = decision["recursion_strength"].cpu()
    correlation = spearmanr(
        val_scores.numpy(),
        decision["probability"].cpu().numpy() @ np.arange(3.0),
    ).statistic
    bins = torch.bucketize(
        val_scores,
        torch.tensor(
            [calibration.low_threshold, calibration.high_threshold],
            dtype=val_scores.dtype,
        ),
    )
    bin_audit = {}
    for bin_index, name in enumerate(("low", "medium", "high")):
        selected = bins == bin_index
        bin_audit[name] = {
            "cases": int(selected.sum()),
            "actions": {
                action_name: int((action[selected] == index).sum())
                for index, action_name in enumerate(POLICY_ACTION_NAMES)
            },
        }
    report = {
        "status": "complete",
        "train_cases": len(train_features),
        "val_cases": len(val_features),
        "calibration": {
            "low_quantile": calibration.low_quantile,
            "high_quantile": calibration.high_quantile,
            "low_threshold": calibration.low_threshold,
            "high_threshold": calibration.high_threshold,
        },
        "metrics": action_metrics(action, val_labels),
        "complexity_decision_audit": bin_audit,
        "score_action_spearman": float(correlation),
        "expected_action_correlation": float(
            np.corrcoef(
                expected_strength.numpy(),
                strength.numpy(),
            )[0, 1]
        ),
        "best_of_n_used": False,
        "target_used_for_inference": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": policy.state_dict(),
            "feature_size": train_features.shape[1],
            "calibration": report["calibration"],
            "metrics": report["metrics"],
            "config": vars(args),
        },
        args.output_dir / "best.pt",
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.report.parent / "recursion_policy_decisions_v1.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "name": name,
                    "complexity": float(score),
                    "expected": POLICY_ACTION_NAMES[int(label)],
                    "predicted": POLICY_ACTION_NAMES[int(prediction)],
                    "recursion_strength": float(value),
                }
            )
            + "\n"
            for name, score, label, prediction, value in zip(
                val_names,
                val_scores.tolist(),
                val_labels.tolist(),
                action.tolist(),
                strength.tolist(),
            )
        ),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
