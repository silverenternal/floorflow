"""Merge shard-produced skv4 JSONL files and images."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/cache"))
    parser.add_argument("--pattern", default="skv4_shard*")
    parser.add_argument("--output", type=Path, default=Path("outputs/cache/skv4_raw"))
    args = parser.parse_args()
    shards = sorted(path for path in args.root.glob(args.pattern) if path.is_dir())
    if not shards:
        raise FileNotFoundError("no skv4 shard directories found")
    args.output.mkdir(parents=True, exist_ok=True)
    image_output = args.output / "images"
    image_output.mkdir(parents=True, exist_ok=True)
    totals = {"train": 0, "val": 0}
    for split in ("train", "val"):
        with (args.output / f"{split}.jsonl").open("w", encoding="utf-8") as target:
            for shard in shards:
                source = shard / f"{split}.jsonl"
                if not source.is_file():
                    continue
                with source.open(encoding="utf-8") as source_file:
                    for line in source_file:
                        if line.strip():
                            target.write(line)
                            totals[split] += 1
    for shard in shards:
        for image in (shard / "images").glob("*"):
            destination = image_output / image.name
            if destination.exists():
                raise FileExistsError(f"duplicate skv4 image: {destination}")
            shutil.copy2(image, destination)
    report = {
        "shards": [str(path) for path in shards],
        "train": totals["train"],
        "val": totals["val"],
        "images": len(list(image_output.glob("*"))),
    }
    (args.output / "merge_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
