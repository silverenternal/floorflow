#!/usr/bin/env python3
"""Create a deterministic case-disjoint split for candidate consensus shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort-dir", type=Path, required=True)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--val-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=205)
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def _bucket(case_index: int, seed: int, modulus: int) -> int:
    digest = hashlib.sha256(f"{seed}:{case_index}".encode()).hexdigest()
    return int(digest[:16], 16) % modulus


def _link(source: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        if target.resolve() != source.resolve():
            raise ValueError(f"refusing to replace existing split shard: {target}")
        return
    os.symlink(source.resolve(), target)


def main() -> None:
    args = parse_args()
    if args.val_every < 2:
        raise ValueError("val_every must be at least two")
    files = sorted(args.cohort_dir.glob("*.npz"))
    if not files:
        raise ValueError("candidate cohort contains no .npz shards")
    args.train_dir.mkdir(parents=True, exist_ok=True)
    args.val_dir.mkdir(parents=True, exist_ok=True)
    train: list[int] = []
    val: list[int] = []
    for path in files:
        case_index = int(np.load(path)["case_index"])
        is_val = _bucket(case_index, args.seed, args.val_every) == 0
        target = (args.val_dir if is_val else args.train_dir) / path.name
        _link(path, target)
        (val if is_val else train).append(case_index)
    if not train or not val:
        raise ValueError("deterministic split produced an empty partition")
    overlap = set(train) & set(val)
    if overlap:
        raise ValueError(f"train/validation overlap: {sorted(overlap)[:8]}")
    report = {
        "version": "chip-candidate-consensus-split-v1",
        "status": "pass",
        "cohort_dir": str(args.cohort_dir),
        "seed": args.seed,
        "val_every": args.val_every,
        "train_cases": train,
        "val_cases": val,
        "train_count": len(train),
        "val_count": len(val),
        "total_count": len(files),
        "disjoint": True,
        "complete": len(train) + len(val) == len(files),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "train_count",
                    "val_count",
                    "disjoint",
                    "complete",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
