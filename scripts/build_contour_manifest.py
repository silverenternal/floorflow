#!/usr/bin/env python3
"""Freeze contour-only checkpoints, artifacts, and replay command."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from floorflow.evaluation.contour_manifest import build_manifest


def _named_path(value: str) -> tuple[str, str]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    return name, path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val")
    parser.add_argument("--cache-root", default="outputs/cache/cubicasa5k_full_v2")
    parser.add_argument("--checkpoint", action="append", type=_named_path, required=True)
    parser.add_argument("--artifact", action="append", type=_named_path, required=True)
    parser.add_argument(
        "--replay-command",
        action="append",
        default=[],
        help="argument of the frozen evaluation command; repeat in order",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_manifest(
        split=args.split,
        checkpoints=dict(args.checkpoint),
        artifacts=dict(args.artifact),
        command=args.replay_command or sys.argv,
        cache_root=args.cache_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
