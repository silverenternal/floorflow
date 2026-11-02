"""Add observable multi-step context to an existing interaction corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from floorflow.data.interaction_context import enrich_sequence_context

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT / "outputs/control_contracts/model_interactions_v2",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports = {}
    for split in ("train", "val"):
        path = args.root / split / "index.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("split") not in {"train", "val"}:
            raise ValueError("interaction enrichment is train/val only")
        reports[split] = enrich_sequence_context(payload["groups"])
        payload["sequence_context"] = reports[split]
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
