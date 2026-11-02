"""Deterministic FloorSet split construction and compact case caching."""

from __future__ import annotations

import random
import re
from collections import defaultdict
from pathlib import Path

from floorflow.chip_flow.floorset import (
    case_from_shard,
    load_floorset_columns,
    shard_block_count,
)

LAYOUTS_PER_SHARD = 112


def _natural_key(path: Path) -> tuple[int, int]:
    worker = re.search(r"worker_(\d+)", str(path.parent))
    layout = re.search(r"layouts_(\d+)", path.name)
    return (
        int(worker.group(1)) if worker else -1,
        int(layout.group(1)) if layout else -1,
    )


def list_legacy_shards(raw_root: str | Path) -> list[Path]:
    """List every legacy shard in deterministic worker/layout order."""

    root = Path(raw_root)
    return sorted(
        root.glob("worker_*/layouts_*.th"),
        key=_natural_key,
    )


def index_legacy_shards(
    paths: list[Path],
    *,
    progress_every: int = 500,
) -> tuple[list[int], list[int]]:
    """Return shard block counts and layout counts without retaining payloads."""

    block_counts: list[int] = []
    layout_counts: list[int] = []
    for index, path in enumerate(paths, start=1):
        columns = load_floorset_columns(path)
        block_counts.append(shard_block_count(columns))
        layout_counts.append(int(columns[1].shape[0]))
        if progress_every and index % progress_every == 0:
            print(f"indexed {index}/{len(paths)} shards", flush=True)
    return block_counts, layout_counts


def select_legacy_splits(
    block_counts: list[int],
    *,
    seed: int = 20090220,
    dev_per_count: int = 100,
    dev_random: int = 2000,
    train_size: int = 50000,
    train_min_blocks: int = 21,
    train_max_blocks: int = 60,
    layouts_per_shard: int = LAYOUTS_PER_SHARD,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], dict]:
    """Select a 12k dev split and 50k training subset deterministically.

    Dev contains ``dev_per_count`` layouts for every represented block count
    plus ``dev_random`` uniformly sampled layouts. Training is sampled from the
    remaining layouts in the configured block-count range.
    """

    rng = random.Random(seed)
    by_count: dict[int, list[int]] = defaultdict(list)
    for shard, count in enumerate(block_counts):
        by_count[count].append(shard)

    selected: set[tuple[int, int]] = set()
    dev: list[tuple[int, int]] = []
    for count in sorted(by_count):
        count_rng = random.Random(seed + count)
        candidates = by_count[count]
        if len(candidates) * layouts_per_shard < dev_per_count:
            raise ValueError(f"not enough layouts for block count {count}")
        for case_index in count_rng.sample(
            range(len(candidates) * layouts_per_shard),
            dev_per_count,
        ):
            shard_index, layout_index = divmod(case_index, layouts_per_shard)
            entry = (candidates[shard_index], layout_index)
            if entry not in selected:
                selected.add(entry)
                dev.append(entry)

    def random_entry() -> tuple[int, int]:
        shard = rng.randrange(len(block_counts))
        return shard, rng.randrange(layouts_per_shard)

    while len(dev) < len(by_count) * dev_per_count + dev_random:
        entry = random_entry()
        if entry not in selected:
            selected.add(entry)
            dev.append(entry)

    eligible_shards = [
        shard
        for shard, count in enumerate(block_counts)
        if train_min_blocks <= count <= train_max_blocks
    ]
    if not eligible_shards:
        raise ValueError("no training shards in the requested block range")
    train: list[tuple[int, int]] = []
    while len(train) < train_size:
        entry = (rng.choice(eligible_shards), rng.randrange(layouts_per_shard))
        if entry not in selected:
            selected.add(entry)
            train.append(entry)

    metadata = {
        "seed": seed,
        "dev_per_count": dev_per_count,
        "dev_random": dev_random,
        "dev_block_counts": {
            str(count): sum(1 for shard, _ in dev if block_counts[shard] == count)
            for count in sorted(by_count)
        },
        "train_size": train_size,
        "train_block_counts": {
            str(count): sum(
                1 for shard, _ in train if block_counts[shard] == count
            )
            for count in range(train_min_blocks, train_max_blocks + 1)
        },
        "train_min_blocks": train_min_blocks,
        "train_max_blocks": train_max_blocks,
        "layouts_per_shard": layouts_per_shard,
    }
    return sorted(train), sorted(dev), metadata


def materialize_cases(
    paths: list[Path],
    train_entries: list[tuple[int, int]],
    dev_entries: list[tuple[int, int]],
    *,
    progress_every: int = 500,
) -> tuple[list[dict], list[dict]]:
    """Load selected layouts once per shard into compact case dictionaries."""

    requested: dict[int, dict[int, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for split, entries in (("train", train_entries), ("dev", dev_entries)):
        for shard, layout_index in entries:
            requested[shard][layout_index].append(split)

    train_by_position = {
        entry: position for position, entry in enumerate(train_entries)
    }
    dev_by_position = {
        entry: position for position, entry in enumerate(dev_entries)
    }
    train_cases: list[dict | None] = [None] * len(train_entries)
    dev_cases: list[dict | None] = [None] * len(dev_entries)

    for done, shard in enumerate(sorted(requested), start=1):
        columns = load_floorset_columns(paths[shard])
        for layout_index in sorted(requested[shard]):
            case = case_from_shard(
                columns,
                layout_index,
                source=str(paths[shard]),
            )
            for split in requested[shard][layout_index]:
                position = (
                    train_by_position[(shard, layout_index)]
                    if split == "train"
                    else dev_by_position[(shard, layout_index)]
                )
                target = train_cases if split == "train" else dev_cases
                target[position] = case
        if progress_every and done % progress_every == 0:
            print(
                f"materialized {done}/{len(requested)} selected shards",
                flush=True,
            )

    if any(case is None for case in train_cases):
        raise RuntimeError("training cache contains missing cases")
    if any(case is None for case in dev_cases):
        raise RuntimeError("dev cache contains missing cases")
    return (
        [case for case in train_cases if case is not None],
        [case for case in dev_cases if case is not None],
    )


__all__ = [
    "LAYOUTS_PER_SHARD",
    "index_legacy_shards",
    "list_legacy_shards",
    "materialize_cases",
    "select_legacy_splits",
]
