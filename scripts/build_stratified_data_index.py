#!/usr/bin/env python3
"""Build a fine-grained, corpus-aware CubiCasa training index."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from floorflow.data.dataset import CubiCasaFloorplanDataset, load_or_rasterize
from floorflow.data.stratified_sampling import compute_floorplan_stratum


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path(".data/cubicasa5k"))
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("outputs/cache/cubicasa5k_full_v2"),
    )
    parser.add_argument("--splits", nargs="+", default=("train", "val", "test"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/data/cubicas_strata_v1.jsonl"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("outputs/reports/cubicas_strata_v1.json"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary = {"splits": {}, "total": 0}
    with args.output.open("w", encoding="utf-8") as handle:
        for split in args.splits:
            dataset = CubiCasaFloorplanDataset(
                args.data_root,
                split,
                image_size=256,
                cache_root=args.cache_root,
                augment=False,
            )
            corpus = Counter()
            strata = Counter()
            fine_strata = Counter()
            present_room_sets = Counter()
            rare = 0
            room_counts = []
            components = []
            for sample in dataset.samples:
                target, footprint = load_or_rasterize(
                    sample,
                    image_size=256,
                    taxonomy=dataset.taxonomy,
                )
                name = f"{sample.relative_case}/{sample.floor_id}"
                entry = compute_floorplan_stratum(
                    name,
                    split,
                    target,
                    footprint,
                    dataset.taxonomy,
                )
                handle.write(json.dumps(entry.to_dict()) + "\n")
                corpus[entry.corpus] += 1
                strata[entry.stratum] += 1
                fine_strata[entry.fine_stratum] += 1
                present_room_sets[",".join(map(str, entry.present_room_classes))] += 1
                rare += int(entry.has_rare)
                room_counts.append(entry.room_class_count)
                components.append(entry.room_component_count)
            summary["splits"][split] = {
                "cases": len(dataset),
                "corpus": dict(corpus),
                "strata": len(strata),
                "fine_strata": len(fine_strata),
                "rare_cases": rare,
                "mean_room_classes": sum(room_counts) / len(room_counts),
                "mean_room_components": sum(components) / len(components),
                "largest_strata": strata.most_common(20),
                "largest_fine_strata": fine_strata.most_common(20),
                "largest_room_class_sets": present_room_sets.most_common(20),
            }
            summary["total"] += len(dataset)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
