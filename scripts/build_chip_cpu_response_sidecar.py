#!/usr/bin/env python3
"""Build a small exact CPU legalizer-response sidecar for quick screening."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party/Trinity/src"))

from floorflow.chip_flow.cache import PackedCaseStore
from floorflow.chip_flow.model import decode_boxes_batch
from floorflow.chip_flow.response_sidecar import (
    build_response_sidecar,
    write_response_sidecar,
)
from floorflow.chip_flow.response_sidecar_selection import select_response_cases
from floorflow.chip_flow.training_manifest import load_train_subset
from scripts.evaluate_chip_official import case_payload, score_case


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache",
        type=Path,
        default=ROOT / "outputs/cache/chip_floorset_50k_train_12k_dev_v2/train",
    )
    parser.add_argument("--train-indices", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-cases", type=int, default=32)
    parser.add_argument("--stratified", action="store_true")
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--legalizer-time-limit-ms", type=int, default=2000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--indices-output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store = PackedCaseStore(args.cache)
    available = sorted(load_train_subset(args.train_indices))
    selected = (
        select_response_cases(
            store,
            available,
            max_cases=args.max_cases,
            seed=args.seed,
        )
        if args.stratified
        else available[: args.max_cases]
    )
    before_rows = {}
    after_rows = {}
    area_rows = {}
    rung_rows = {}
    for count in sorted({int(store.arrays["n"][index]) for index in selected}):
        group = [index for index in selected if int(store.arrays["n"][index]) == count]
        for start in range(0, len(group), args.batch_size):
            indices = group[start : start + args.batch_size]
            batch = store.batch(np.asarray(indices))
            batch["_indices"] = torch.as_tensor(indices)
            before = decode_boxes_batch(batch["source_latent"], batch["areas"])
            for row, index in enumerate(indices):
                scored = score_case(
                    case_payload(
                        batch,
                        row,
                        before[row],
                        0.0,
                        legalize=True,
                        legalizer_time_limit_ms=args.legalizer_time_limit_ms,
                    )
                    | {
                        "return_legalized_boxes": True,
                    }
                )
                before_rows[index] = before[row, :count].clone()
                after_rows[index] = torch.as_tensor(
                    scored["legalized_xywh"],
                    dtype=torch.float32,
                )
                area_rows[index] = batch["areas"][row, :count].clone()
                rung_rows[index] = int(scored["legalizer_rung"])
    ordered = [index for index in selected if index in before_rows]
    lengths = torch.tensor(
        [before_rows[index].shape[0] for index in ordered],
        dtype=torch.int16,
    )
    maximum = int(lengths.max())
    before = torch.zeros(len(ordered), maximum, 4)
    after = torch.zeros_like(before)
    areas = torch.zeros(len(ordered), maximum)
    for row, index in enumerate(ordered):
        length = int(lengths[row])
        before[row, :length] = before_rows[index]
        after[row, :length] = after_rows[index]
        areas[row, :length] = area_rows[index]
    rung = torch.tensor([rung_rows[index] for index in ordered])
    sidecar = build_response_sidecar(
        np.asarray(ordered, dtype=np.int64),
        before,
        after,
        areas,
        rung,
        lengths=lengths,
        metadata={
            "source": "shelf source_latent through CPU scale-pack",
            "cache": str(args.cache),
            "train_indices": str(args.train_indices),
        },
    )
    write_response_sidecar(args.output, sidecar)
    displacement = torch.from_numpy(sidecar.displacement)
    valid = (
        torch.arange(displacement.shape[1])[None]
        < torch.from_numpy(sidecar.lengths)[:, None]
    )
    displacement_l2 = displacement.norm(dim=-1)
    report = {
        "status": "complete_cpu_only",
        "cases": len(ordered),
        "selection": "stratified_complexity" if args.stratified else "prefix",
        "selection_seed": int(args.seed),
        "indices": ordered,
        "rung_counts": {
            str(int(value)): int(rung.eq(value).sum())
            for value in torch.unique(rung)
        },
        "displacement_l2_mean": float(
            displacement_l2[valid].mean()
        ),
        "displacement_l2_p95": float(
            displacement_l2[valid].quantile(0.95)
        ),
        "output": str(args.output),
        "boundary": "Shelf-source quick-screen labels; not a deployable candidate sidecar.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.indices_output is not None:
        args.indices_output.parent.mkdir(parents=True, exist_ok=True)
        args.indices_output.write_text(
            json.dumps({"indices": ordered}, indent=2) + "\n",
            encoding="utf-8",
        )
    print(args.report)


if __name__ == "__main__":
    main()
