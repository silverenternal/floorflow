#!/usr/bin/env python3
"""Fit a CPU portfolio router from reports with pre-generation router features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from floorflow.chip_flow.cache import PackedCaseStore
from floorflow.chip_flow.portfolio_router import (
    PROFILE_NAMES,
    PortfolioRouter,
    router_features,
)
from floorflow.chip_flow.training_runtime import (
    save_torch_atomic,
    write_json_atomic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument(
        "--cache",
        type=Path,
        default=ROOT / "outputs/cache/chip_official_test_100_v1",
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=19)
    return parser.parse_args()


def _rows(
    paths: list[Path],
    store: PackedCaseStore,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features = []
    labels = []
    case_indices = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("rows", []):
            values = row.get("router_features")
            if values is None:
                batch = store.batch(
                    [int(row["index"])],
                )
                values = router_features(
                    batch["constraints"],
                    batch["b2b"],
                    batch["p2b"],
                    batch["n_b2b"],
                    batch["n_p2b"],
                    batch["areas"],
                    batch["pins"],
                    batch["n_pins"],
                )[0].tolist()
            features.append(values)
            labels.append(PROFILE_NAMES.index(str(row["selection"])))
            case_indices.append(int(row["index"]))
    if not features:
        raise ValueError("reports contain no router_features rows")
    return (
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(case_indices, dtype=torch.long),
    )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    features, labels, case_indices = _rows(
        args.reports,
        PackedCaseStore(args.cache),
    )
    validation_mask = case_indices.remainder(5).eq(0)
    train_mask = ~validation_mask
    train_features, train_labels = features[train_mask], labels[train_mask]
    val_features, val_labels = features[validation_mask], labels[validation_mask]
    if len(torch.unique(train_labels)) < 2:
        raise ValueError("router fitting requires at least two selected profiles")
    model = PortfolioRouter(args.hidden)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    history = []
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_features)
        loss = F.cross_entropy(logits, train_labels)
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % 100 == 0 or step + 1 == args.steps:
            detached = logits.detach()
            val_logits = model(val_features)
            history.append(
                {
                    "step": step + 1,
                    "train_loss": float(loss.detach()),
                    "train_accuracy": float(
                        detached.argmax(dim=-1).eq(train_labels).float().mean()
                    ),
                    "validation_accuracy": float(
                        val_logits.argmax(dim=-1).eq(val_labels).float().mean()
                    ),
                }
            )
    save_torch_atomic(
        args.output,
        {
            "model": model.state_dict(),
            "config": {
                "hidden": args.hidden,
                "feature_dim": 10,
                "profiles": PROFILE_NAMES,
                "validation_accuracy": max(
                    row["validation_accuracy"] for row in history
                ),
                "split": "case_index modulo 5 holdout",
            },
            "fitting": {
                "cases": len(labels),
                "train_cases": len(train_labels),
                "validation_cases": len(val_labels),
                "split": "case_index modulo 5 holdout",
                "reports": [str(path) for path in args.reports],
                "seed": args.seed,
            },
        },
    )
    write_json_atomic(
        args.report,
        {
            "status": "fit_complete_cpu_only",
            "cases": len(labels),
            "train_cases": len(train_labels),
            "validation_cases": len(val_labels),
            "split": "case_index modulo 5 holdout",
            "profile_counts": {
                name: int(labels.eq(index).sum())
                for index, name in enumerate(PROFILE_NAMES)
            },
            "history": history,
            "best_validation_accuracy": max(
                row["validation_accuracy"] for row in history
            ),
            "checkpoint": str(args.output),
            "boundary": "Router fitting on already observed rows is development evidence only.",
        },
    )
    print(args.report)


if __name__ == "__main__":
    main()
