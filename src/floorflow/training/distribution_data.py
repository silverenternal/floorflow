"""Dataset construction shared by distribution-CFM training sources."""

from __future__ import annotations

from torch.utils.data import ConcatDataset

from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.data.skv4_dataset import Skv4PlanDataset
from floorflow.data.taxonomy import FloorplanTaxonomy


def distribution_dataset(
    args,
    split: str,
    taxonomy: FloorplanTaxonomy,
    *,
    image_size: int,
    augment: bool,
    limit: int | None,
):
    if args.dataset == "cubicasa":
        return CubiCasaFloorplanDataset(
            args.data_root,
            split,
            image_size=image_size,
            cache_root=args.cache_root,
            augment=augment,
            limit=limit,
        )
    if args.dataset == "skv4":
        return Skv4PlanDataset(
            args.skv4_root,
            split,
            taxonomy,
            image_size=image_size,
            augment=augment,
            limit=limit,
        )
    if args.dataset == "mixed":
        if split != "train":
            return CubiCasaFloorplanDataset(
                args.data_root,
                split,
                image_size=image_size,
                cache_root=args.cache_root,
                augment=False,
                limit=limit,
            )
        source_limit = None if limit is None else max(1, int(limit) // 2)
        return ConcatDataset(
            [
                CubiCasaFloorplanDataset(
                    args.data_root,
                    "train",
                    image_size=image_size,
                    cache_root=args.cache_root,
                    augment=augment,
                    limit=source_limit,
                ),
                Skv4PlanDataset(
                    args.skv4_root,
                    "train",
                    taxonomy,
                    image_size=image_size,
                    augment=augment,
                    limit=source_limit,
                ),
            ]
        )
    raise ValueError(f"unsupported distribution dataset: {args.dataset}")


__all__ = ["distribution_dataset"]
