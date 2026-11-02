#!/usr/bin/env python3
"""Build a train-only room-class adjacency prior."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.representation.room_pair_prior import (
    build_room_pair_prior,
    write_room_pair_prior,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("outputs/cache/cubicasa5k_full_v2"),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    taxonomy = FloorplanTaxonomy.read(args.cache_root / "taxonomy.json")
    payload = build_room_pair_prior(
        args.cache_root,
        taxonomy,
        limit=args.limit,
    )
    write_room_pair_prior(payload, args.output)
    print(
        json.dumps(
            {
                "cases": payload["cases"],
                "fingerprint": payload["fingerprint"],
                "shape": [
                    len(payload["room_ids"]),
                    len(payload["room_ids"]),
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
