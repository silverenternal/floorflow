#!/usr/bin/env python3
"""Audit dynamic sparse-routing behavior before model training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from floorflow.chip_flow.cache import PackedCaseStore
from floorflow.chip_flow.sparse_hierarchy import SparseHierarchyEncoder

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache",
        type=Path,
        default=ROOT / "outputs/cache/chip_official_test_100_v1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_sparse_hierarchy_preflight_v1.json",
    )
    parser.add_argument("--max-slots", type=int, default=8)
    parser.add_argument("--min-slots", type=int, default=2)
    parser.add_argument("--assignment-top-k", type=int, default=2)
    parser.add_argument("--seed", type=int, default=19)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    store = PackedCaseStore(args.cache)
    model = SparseHierarchyEncoder(
        feature_dim=int(store.metadata["feature_dim"]),
        hidden=32,
        layers=1,
        heads=4,
        max_slots=args.max_slots,
        min_slots=args.min_slots,
        assignment_top_k=args.assignment_top_k,
    ).eval()
    indices = store.indices()
    counts = []
    complexities = []
    routing_edges = 0
    rows = []
    for count in sorted(set(store.arrays["n"][indices].tolist())):
        batch_indices = indices[store.arrays["n"][indices] == count][:4]
        batch = store.batch(batch_indices, device="cpu")
        result = model(
            batch["features"],
            b2b=batch["b2b"],
            p2b=batch["p2b"],
            pins=batch["pins"],
            n_b2b=batch["n_b2b"],
            n_p2b=batch["n_p2b"],
            n_pins=batch["n_pins"],
        )
        active = result.diagnostics["active_slot_count"].tolist()
        complexity = result.diagnostics["complexity"].tolist()
        counts.extend(active)
        complexities.extend(complexity)
        routing_edges += len(active) * count * args.assignment_top_k
        rows.append(
            {
                "blocks": count,
                "cases": len(active),
                "active_slots": active,
                "complexity": complexity,
            }
        )
    histogram = {
        str(value): counts.count(value)
        for value in sorted(set(counts))
    }
    report = {
        "status": "pass" if len(set(counts)) > 1 else "fail",
        "scope": "random-init routing preflight; not a quality or accuracy result",
        "cache": str(args.cache),
        "cases": len(counts),
        "active_slot_count": {
            "min": min(counts),
            "max": max(counts),
            "mean": float(sum(counts) / len(counts)),
            "unique": sorted(set(counts)),
            "histogram": histogram,
        },
        "complexity": {
            "min": min(complexities),
            "max": max(complexities),
            "mean": float(sum(complexities) / len(complexities)),
        },
        "assignment_top_k": args.assignment_top_k,
        "routing_edges": routing_edges,
        "rows": rows,
        "best_of_n_used": False,
        "target_derived_control": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["active_slot_count"], indent=2))


if __name__ == "__main__":
    main()
