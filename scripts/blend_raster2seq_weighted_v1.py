"""Blend two trusted Raster2Seq checkpoints for validation-only model soup."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from blend_raster2seq_checkpoints_v1 import blend_state_dicts

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--secondary", type=Path, required=True)
    parser.add_argument("--secondary-weight", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.secondary_weight <= 1.0:
        raise ValueError("secondary weight must be in [0, 1]")
    primary = torch.load(args.primary, map_location="cpu", weights_only=False)
    secondary = torch.load(args.secondary, map_location="cpu", weights_only=False)
    state = blend_state_dicts(
        primary["ema"],
        secondary["ema"],
        args.secondary_weight,
        mask_from_secondary=False,
    )
    payload = {
        key: value
        for key, value in primary.items()
        if key not in {"model", "ema", "optimizer", "lr_scheduler"}
    }
    payload.update(
        {
            "model": state,
            "ema": state,
            "blend_primary": str(args.primary),
            "blend_secondary": str(args.secondary),
            "blend_secondary_weight": float(args.secondary_weight),
        }
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    report = {
        "status": "complete",
        "primary": str(args.primary),
        "secondary": str(args.secondary),
        "secondary_weight": float(args.secondary_weight),
        "output": str(output.relative_to(ROOT)),
        "output_sha256": sha256(output),
        "selection_split": "val",
        "best_of_n_used": False,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
