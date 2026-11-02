"""Room-count and complexity-aware sampling for instance-head training."""

from __future__ import annotations

import random

from torch.utils.data import DataLoader

from floorflow.data.dataset import filter_unsupervised_samples
from floorflow.data.stratified_sampling import (
    CurriculumStratifiedSampler,
    compute_floorplan_stratum,
)


def prepare_raster_dataset(
    dataset,
    *,
    limit: int | None,
    offset: int = 0,
    shuffle_seed: int | None = None,
) -> list[str]:
    """Filter empty targets, then select one deterministic, non-overlapping slice."""

    if offset < 0:
        raise ValueError("dataset offset must be non-negative")
    if limit is not None and limit < 0:
        raise ValueError("dataset limit must be non-negative")
    removed = filter_unsupervised_samples(dataset)
    selected = list(dataset.samples)
    if shuffle_seed is not None:
        random.Random(int(shuffle_seed)).shuffle(selected)
    stop = None if limit is None else int(offset) + int(limit)
    selected = selected[int(offset) : stop]
    if limit is not None and len(selected) != int(limit):
        raise ValueError(
            f"requested {limit} samples at offset {offset}, "
            f"but only {len(selected)} valid samples remain"
        )
    dataset.samples = selected
    dataset._targets = None
    dataset._footprints = None
    return removed


def raster_dataset_kwargs(args) -> dict:
    """Return dataset-specific kwargs without duplicating mode alignment."""

    common = {
        "include_room_targets": not bool(args.teacher_only),
        "room_target_limit": int(args.queries),
    }
    if args.control_raster_root is not None:
        return {
            "control_root": args.control_raster_root,
            "page_floor_crop": bool(args.page_floor_crop),
            **common,
        }
    return {
        "image_mode": "floor" if args.page_floor_crop else "native",
        **common,
    }


def build_room_curriculum_sampler(
    dataset,
    taxonomy,
    *,
    num_samples: int,
    seed: int,
    rare_boost: float = 2.0,
    curriculum_strength: float = 1.0,
):
    dataset.preload()
    if dataset._targets is None or dataset._footprints is None:
        raise RuntimeError("dataset preload did not materialize room targets")
    entries = []
    for index, sample in enumerate(dataset.samples):
        name = f"{sample.relative_case}__{sample.floor_id}"
        entries.append(
            compute_floorplan_stratum(
                name,
                "train",
                dataset._targets[index],
                dataset._footprints[index],
                taxonomy,
            )
        )
    return CurriculumStratifiedSampler(
        entries,
        num_samples=int(num_samples),
        rare_boost=float(rare_boost),
        curriculum_strength=float(curriculum_strength),
        seed=int(seed),
    )


def build_room_loaders(
    args,
    train_dataset,
    val_dataset,
    taxonomy,
):
    loader_args = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": str(args.device).startswith("cuda"),
        "persistent_workers": args.workers > 0,
    }
    if args.workers > 0:
        loader_args["prefetch_factor"] = int(
            getattr(args, "prefetch_factor", 4)
        )
    if args.train_sampling == "curriculum":
        sampler = build_room_curriculum_sampler(
            train_dataset,
            taxonomy,
            num_samples=args.limit_train,
            seed=args.seed,
            rare_boost=args.curriculum_rare_boost,
            curriculum_strength=args.curriculum_strength,
        )
        train_loader = DataLoader(
            train_dataset,
            sampler=sampler,
            **loader_args,
        )
    else:
        sampler = None
        train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            **loader_args,
        )
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_args)
    return train_loader, val_loader, sampler


__all__ = [
    "build_room_curriculum_sampler",
    "build_room_loaders",
    "prepare_raster_dataset",
    "raster_dataset_kwargs",
]
