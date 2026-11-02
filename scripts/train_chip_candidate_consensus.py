#!/usr/bin/env python3
"""Distill ranked M75 candidates into a single consensus fuser."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from floorflow.chip_flow.candidate_consensus import CandidateConsensusFuser
from floorflow.chip_flow.candidate_consensus_training import (
    candidate_consensus_objective,
    load_candidate_consensus_batch,
)

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--teacher-temperature", type=float, default=0.1)
    parser.add_argument("--endpoint-weight", type=float, default=1.0)
    parser.add_argument("--teacher-weight", type=float, default=0.1)
    parser.add_argument("--entropy-weight", type=float, default=0.001)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument(
        "--selection-mode",
        choices=("soft", "topk_residual"),
        default="soft",
    )
    parser.add_argument("--residual-limit", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1946)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _files(root: Path) -> list[Path]:
    files = sorted(root.glob("*.npz"))
    if not files:
        raise ValueError("candidate cohort directory contains no .npz shards")
    return files


def _meta(files: list[Path]) -> tuple[int, int]:
    meta = []
    for path in files:
        sample = np.load(path)
        meta.append(
            (
                int(sample["candidate_features"].shape[-1]),
                int(sample["context"].shape[-1])
                if "context" in sample.files
                else 0,
            )
        )
    if len(set(meta)) != 1:
        raise ValueError("candidate cohort shards disagree on feature/context width")
    return meta[0]


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.lr <= 0.0 or args.grad_clip < 0.0:
        raise ValueError("invalid training budget")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    files = _files(args.cohort_dir)
    feature_dim, context_dim = _meta(files)
    top_k = args.top_k or None
    model = CandidateConsensusFuser(
        args.hidden,
        candidate_feature_dim=feature_dim,
        context_dim=context_dim,
        top_k=top_k,
        selection_mode=args.selection_mode,
        residual_limit=args.residual_limit,
    ).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    history = []
    order = np.arange(len(files))
    for epoch in range(args.epochs):
        np.random.shuffle(order)
        totals = []
        for index in order:
            batch = load_candidate_consensus_batch(files[int(index)])
            batch = type(batch)(
                candidate_states=batch.candidate_states.to(args.device),
                candidate_features=batch.candidate_features.to(args.device),
                quality=batch.quality.to(args.device),
                target_state=batch.target_state.to(args.device),
                candidate_mask=batch.candidate_mask.to(args.device),
                context=(
                    batch.context.to(args.device)
                    if batch.context is not None
                    else None
                ),
            )
            loss, _weights, terms = candidate_consensus_objective(
                model,
                batch,
                teacher_temperature=args.teacher_temperature,
                endpoint_weight=args.endpoint_weight,
                teacher_weight=args.teacher_weight,
                entropy_weight=args.entropy_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            totals.append(
                {
                    key: float(value.detach().cpu())
                    for key, value in terms.as_dict().items()
                }
            )
        history.append(
            {
                "epoch": epoch + 1,
                **{
                    key: float(np.mean([row[key] for row in totals]))
                    for key in totals[0]
                },
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        **vars(args),
        "top_k": top_k,
        "cohort_dir": str(args.cohort_dir),
        "output_dir": str(args.output_dir),
        "feature_dim": feature_dim,
        "context_dim": context_dim,
        "contract": "quality is lower-is-better; candidate 0 is the identity reference",
    }
    payload = {
        "model": model.state_dict(),
        "config": json.loads(json.dumps(config, default=str)),
        "history": history,
    }
    torch.save(payload, args.output_dir / "final.pt")
    (args.output_dir / "report.json").write_text(
        json.dumps(
            {**payload, "model": "stored in final.pt", "status": "complete"},
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
