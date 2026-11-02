"""Build a two-step legal-recovery corpus from real predicted controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from floorflow.data.interaction_candidate_dataset import (
    build_interaction_groups,
    write_interaction_index,
)
from floorflow.data.taxonomy import FloorplanTaxonomy

ROOT = Path(__file__).resolve().parents[1]
ORACLE_ROOTS = {
    "train": "outputs/control_contracts/oracle_v1",
    "val": "outputs/control_contracts/oracle_cagn_v1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--steps", type=int, choices=(2, 3), default=2)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--predicted-root",
        type=Path,
        default=ROOT / "outputs/control_contracts/model_predicted_v1",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs/control_contracts/model_interactions_v1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(max(1, args.threads))
    taxonomy = FloorplanTaxonomy.read(
        ROOT / "outputs/cache/cubicasa5k_full_v2/taxonomy.json"
    )
    oracle_root = ROOT / ORACLE_ROOTS[args.split]
    groups = build_interaction_groups(
        args.predicted_root,
        args.split,
        taxonomy,
        target_cache_root=ROOT / "outputs/cache/cubicasa5k",
        oracle_control_root=oracle_root,
        case_limit=args.limit,
        steps=args.steps,
    )
    output = args.output_root / args.split / "index.json"
    payload = write_interaction_index(
        output,
        groups,
        split=args.split,
        predicted_root=str(args.predicted_root.relative_to(ROOT)),
        oracle_control_root=str(oracle_root.relative_to(ROOT)),
    )
    print(
        json.dumps(
            {
                "output": (
                    str(output.relative_to(ROOT))
                    if output.is_relative_to(ROOT)
                    else str(output)
                ),
                "split": args.split,
                **payload["counts"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
