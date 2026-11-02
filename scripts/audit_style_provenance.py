#!/usr/bin/env python3
"""Audit train-only style labels and explicit deployment-time conditioning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from floorflow.data.dataset import collect_samples
from floorflow.evaluation.style_provenance import audit_style_provenance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--style-bank", type=Path, required=True)
    parser.add_argument("--dataset", choices=("cubicasa", "skv4"), default="cubicasa")
    parser.add_argument("--data-root", type=Path, default=Path(".data/cubicasa5k"))
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--skv4-root", type=Path, default=Path("outputs/cache/skv4"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def split_names(args: argparse.Namespace, split: str) -> list[str]:
    if args.dataset == "skv4":
        names = []
        with (args.skv4_root / f"{split}.jsonl").open(encoding="utf-8") as source:
            for line in source:
                names.append(str(json.loads(line)["id"]))
        return names
    return [
        f"{sample.relative_case}__{sample.floor_id}"
        for sample in collect_samples(
            args.data_root,
            split,
            cache_root=args.cache_root,
        )
    ]


def main() -> None:
    args = parse_args()
    bank = json.loads(args.style_bank.read_text(encoding="utf-8"))
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    manifest = json.loads(args.run_manifest.read_text(encoding="utf-8"))
    report = audit_style_provenance(
        bank,
        train_names=split_names(args, "train"),
        val_names=split_names(args, "val"),
        test_names=split_names(args, "test"),
        checkpoint=checkpoint,
        run_manifest=manifest,
    )
    report.update(
        {
            "style_bank": str(args.style_bank),
            "checkpoint": str(args.checkpoint),
            "run_manifest": str(args.run_manifest),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
