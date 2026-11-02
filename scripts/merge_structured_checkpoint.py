#!/usr/bin/env python3
"""Merge a frozen-backbone structured-object head into a full checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--object-head", type=Path, required=True)
    parser.add_argument("--room-program-head", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preserve-base-wall", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = torch.load(args.base, map_location="cpu", weights_only=False)
    object_checkpoint = torch.load(args.object_head, map_location="cpu", weights_only=False)
    if base["taxonomy"] != object_checkpoint["taxonomy"]:
        raise ValueError("base and object-head taxonomies differ")
    head = {
        f"structured_objects.{key}": value
        for key, value in object_checkpoint["head"].items()
        if not (
            args.preserve_base_wall
            and key.startswith("wall.")
        )
    }
    if args.room_program_head is not None:
        program_checkpoint = torch.load(args.room_program_head, map_location="cpu", weights_only=False)
        if object_checkpoint["taxonomy"] != program_checkpoint["taxonomy"]:
            raise ValueError("room-program and object-head taxonomies differ")
        head.update(
            {
                f"room_program.{key}": value
                for key, value in program_checkpoint["head"].items()
            }
        )
        base.setdefault("config", {})["room_program_checkpoint"] = str(args.room_program_head)
    for state_name in ("model", "ema_model", "raw_model"):
        state = base.get(state_name)
        if state is None:
            continue
        state.update(head)
    base.setdefault("config", {})["structured_object_checkpoint"] = str(args.object_head)
    base["structured_object_config"] = object_checkpoint.get("config", {})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(base, args.output)
    print(json.dumps({"output": str(args.output), "merged_tensors": len(head)}, indent=2))


if __name__ == "__main__":
    main()
