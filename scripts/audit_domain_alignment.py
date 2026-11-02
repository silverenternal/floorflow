#!/usr/bin/env python3
"""Compare CubiCasa and synthetic class distributions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np

from floorflow.data.domain_alignment import (
    class_alignment_rows,
    distribution_summary,
    normalized_pixel_distribution,
)
from floorflow.data.taxonomy import FloorplanTaxonomy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--real-cache",
        type=Path,
        default=REPO_ROOT / "outputs/cache/cubicasa5k_full_v2",
    )
    parser.add_argument(
        "--synthetic-cache",
        type=Path,
        default=REPO_ROOT / "outputs/cache/fpgen_pretrain2k_v1",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    taxonomy = FloorplanTaxonomy.read(args.real_cache / "taxonomy.json")
    real_frequency = json.loads(
        (args.real_cache / "class_frequency.json").read_text(encoding="utf-8")
    )
    synthetic_manifest = json.loads(
        (args.synthetic_cache / "manifest.json").read_text(encoding="utf-8")
    )
    if tuple(real_frequency["class_names"]) != tuple(synthetic_manifest["class_names"]):
        raise ValueError("real and synthetic taxonomy ordering differs")
    real_counts = np.asarray(real_frequency["pixel_counts"], dtype=np.float64)
    synthetic_counts = np.asarray(synthetic_manifest["class_pixels"], dtype=np.float64)
    room_ids = list(taxonomy.room_ids)
    functional_ids = list(taxonomy.room_ids) + list(taxonomy.furniture_ids)
    room_real = normalized_pixel_distribution(real_counts, room_ids)
    room_synthetic = normalized_pixel_distribution(synthetic_counts, room_ids)
    functional_real = normalized_pixel_distribution(real_counts, functional_ids)
    functional_synthetic = normalized_pixel_distribution(
        synthetic_counts,
        functional_ids,
    )
    rows = class_alignment_rows(
        taxonomy.class_names,
        functional_ids,
        real_counts,
        synthetic_counts,
    )
    report = {
        "real_cache": str(args.real_cache),
        "synthetic_cache": str(args.synthetic_cache),
        "room_distribution": distribution_summary(room_real, room_synthetic),
        "functional_distribution": distribution_summary(
            functional_real,
            functional_synthetic,
        ),
        "missing_real_classes": [
            row["class"] for row in rows if row["missing_in_synthetic"]
        ],
        "unsupported_synthetic_classes": [
            row["class"] for row in rows if row["missing_in_real"]
        ],
        "largest_mismatches": rows[:20],
        "all_rows": rows,
        "mixed_training_gate": {
            "allowed": False,
            "reason": "class mapping and mixed-batch domain probe must be rerun after this audit",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "room_distribution": report["room_distribution"],
                "functional_distribution": report["functional_distribution"],
                "missing_real_class_count": len(report["missing_real_classes"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
