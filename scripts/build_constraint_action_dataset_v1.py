"""Build train/val constraint-action manifests without reading test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from floorflow.data.constraint_action_dataset import (
    build_action_records,
)
from floorflow.data.constraint_candidate_dataset import write_candidate_index
from floorflow.data.taxonomy import FloorplanTaxonomy

ROOT = Path(__file__).resolve().parents[1]
SPLIT_ROOTS = {
    "train": "outputs/control_contracts/oracle_v1",
    "val": "outputs/control_contracts/oracle_cagn_v1",
    "test": "outputs/control_contracts/oracle_cagn_test_v1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=tuple(SPLIT_ROOTS), default="train")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--seed", type=int, default=2300)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/control_contracts/residential_cagn_v1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    taxonomy = FloorplanTaxonomy.read(
        ROOT / "outputs/cache/cubicasa5k_full_v2/taxonomy.json"
    )
    control_root = ROOT / SPLIT_ROOTS[args.split]
    records = build_action_records(
        control_root,
        args.split,
        taxonomy,
        seed=args.seed,
        case_limit=args.limit,
        allow_test=args.allow_test,
    )
    output = (args.output / args.split / "index.json").resolve()
    payload = write_candidate_index(
        output,
        records,
        split=args.split,
        control_root=str(control_root.relative_to(ROOT)),
    )
    print(
        json.dumps(
            {
                "output": str(output.relative_to(ROOT))
                if output.is_relative_to(ROOT)
                else str(output),
                "split": args.split,
                "groups": payload["counts"]["groups"],
                "candidates": payload["counts"]["candidates"],
                "counts": payload["counts"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
