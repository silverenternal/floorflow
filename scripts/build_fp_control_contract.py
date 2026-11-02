#!/usr/bin/env python3
"""Materialize and validate the FP-C01 serialized control contract."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from floorflow.data.control_cache import (
    CachedControlDataset,
    control_cache_path,
    read_control_cache,
    write_control_cache,
)
from floorflow.data.control_contract import (
    CONTROL_SCHEMA_VERSION,
    control_provenance,
    oracle_control_payload,
    sha256_file,
    summarize_control_file,
    validate_control_payload,
)
from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.evaluation.room_metric_suite import room_metric_suite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path(".data/cubicasa5k"))
    parser.add_argument(
        "--dataset-cache-root",
        type=Path,
        default=Path("outputs/cache/cubicasa5k_full_v2"),
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--source",
        choices=("oracle", "predicted", "external"),
        default="oracle",
    )
    parser.add_argument("--input-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--room-size", type=int, default=224)
    parser.add_argument("--structure-size", type=int, default=128)
    parser.add_argument("--furniture-size", type=int, default=192)
    parser.add_argument("--support-size", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def binary_iou(prediction: torch.Tensor, target: torch.Tensor) -> float:
    prediction = prediction.bool()
    target = target.bool()
    union = torch.logical_or(prediction, target).sum()
    if int(union) == 0:
        return 1.0
    return float(torch.logical_and(prediction, target).sum() / union)


def resize_binary(value: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return (
        F.interpolate(
            value[None, None].float(),
            size=size,
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        >= 0.5
    )


def resize_nearest(value: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    batched = value[None].float() if value.ndim == 3 else value[None, None].float()
    return F.interpolate(
        batched,
        size=size,
        mode="nearest",
    )[0]


def control_information(
    target: torch.Tensor,
    footprint: torch.Tensor,
    controls: dict[str, torch.Tensor],
    taxonomy,
) -> dict[str, float]:
    room_ids = torch.tensor(taxonomy.room_ids, device=target.device)
    opening_ids = torch.tensor(taxonomy.opening_ids, device=target.device)
    furniture_ids = torch.tensor(taxonomy.furniture_ids, device=target.device)
    output_size = target.shape[-2:]

    logits = F.interpolate(
        controls["room_logits"][None].float(),
        size=output_size,
        mode="bilinear",
        align_corners=False,
    )[0]
    occupied = logits.max(dim=0).values > 0.0
    control_rooms = room_ids[logits.argmax(dim=0)]
    control_rooms = control_rooms * occupied.long() * footprint.long()
    predicted_room_support = torch.isin(control_rooms, room_ids)
    target_room_support = torch.isin(target, room_ids)
    rooms = room_metric_suite(
        control_rooms.cpu().numpy(),
        target.cpu().numpy(),
        footprint.cpu().numpy(),
        taxonomy,
        predicted_room_support=predicted_room_support.cpu().numpy(),
        predicted_instance_support=predicted_room_support.cpu().numpy(),
        target_room_support=target_room_support.cpu().numpy(),
    )

    room_count = len(taxonomy.room_ids)
    structure = controls["structure"]
    wall = resize_binary(structure[room_count], output_size)
    opening = resize_binary(structure[room_count + 1], output_size)
    furniture_logits = resize_nearest(
        controls["furniture"][room_count:-1],
        output_size,
    )
    furniture_present = furniture_logits.max(dim=0).values > 0.5
    furniture_class = furniture_ids[furniture_logits.argmax(dim=0)]
    control_furniture = (furniture_class * furniture_present.long()) * footprint.long()
    program_target = torch.zeros_like(controls["program"])
    for index, class_id in enumerate(room_ids):
        program_target[index] = torch.isin(target, class_id).any()

    target_support = torch.isin(target, room_ids)
    return {
        "room_macro_iou": float(rooms["visible_semantic"]["room_macro_iou"]),
        "room_support_iou": binary_iou(predicted_room_support, target_support),
        "program_f1": float(
            2
            * torch.logical_and(program_target, controls["program"]).sum()
            / (
                program_target.sum()
                + controls["program"].sum()
            ).clamp_min(1)
        ),
        "wall_iou": binary_iou(wall, target == taxonomy.class_id("wall")),
        "opening_iou": binary_iou(opening, torch.isin(target, opening_ids)),
        "furniture_iou": binary_iou(
            torch.isin(control_furniture, furniture_ids),
            torch.isin(target, furniture_ids),
        ),
    }


def materialize_oracle(
    args: argparse.Namespace,
    dataset: CubiCasaFloorplanDataset,
    loader: DataLoader,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for target, footprint, _floor_ids, names in loader:
        target = target.to(args.device)
        footprint = footprint.to(args.device)
        controls = oracle_control_payload(
            target,
            footprint,
            dataset.taxonomy,
            room_size=args.room_size,
            structure_size=args.structure_size,
            furniture_size=args.furniture_size,
            support_size=args.support_size,
        )
        for index, name in enumerate(names):
            path = control_cache_path(args.output_root, args.split, name)
            if path.exists() and not args.overwrite:
                raise FileExistsError(
                    f"{path} exists; pass --overwrite to rebuild the cache"
                )
            sample_controls = {
                key: value[index : index + 1] for key, value in controls.items()
            }
            write_control_cache(
                path,
                room_logits=sample_controls["room_logits"],
                program=sample_controls["program"],
                structure=sample_controls["structure"],
                furniture=sample_controls["furniture"],
                support=sample_controls["support"],
                room_size=args.room_size,
            )
            information = control_information(
                target[index],
                footprint[index],
                {key: value[index] for key, value in controls.items()},
                dataset.taxonomy,
            )
            rows.append(
                {
                    "name": name,
                    "file": str(path),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                    "information": information,
                }
            )
        print(f"materialized {len(rows)}/{len(dataset)}", flush=True)
    return rows


def index_existing(
    args: argparse.Namespace,
    dataset: CubiCasaFloorplanDataset,
) -> list[dict[str, object]]:
    if args.input_root is None:
        raise ValueError("--input-root is required for non-oracle sources")
    rows: list[dict[str, object]] = []
    for sample in dataset.samples:
        name = f"{sample.relative_case}__{sample.floor_id}"
        path = control_cache_path(args.input_root, args.split, name)
        if not path.is_file():
            raise FileNotFoundError(path)
        summary = summarize_control_file(
            path,
            dataset.taxonomy,
            room_size=args.room_size,
            structure_size=args.structure_size,
            furniture_size=args.furniture_size,
            support_size=args.support_size,
        )
        rows.append(
            {
                "name": name,
                "file": summary.pop("path"),
                **summary,
            }
        )
    return rows


def measure_existing(
    args: argparse.Namespace,
    dataset: CubiCasaFloorplanDataset,
) -> dict[str, dict[str, float]]:
    control_dataset = CachedControlDataset(
        dataset,
        args.input_root,
        args.split,
    )
    loader = DataLoader(
        control_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    measurements: dict[str, dict[str, float]] = {}
    for target, footprint, _floor_ids, names, controls in loader:
        target = target.to(args.device)
        footprint = footprint.to(args.device)
        controls = {key: value.to(args.device) for key, value in controls.items()}
        for index, name in enumerate(names):
            measurements[name] = control_information(
                target[index],
                footprint[index],
                {key: value[index] for key, value in controls.items()},
                dataset.taxonomy,
            )
    return measurements


def build_manifest(
    args: argparse.Namespace,
    dataset: CubiCasaFloorplanDataset,
    rows: list[dict[str, object]],
) -> dict[str, object]:
    provenance = control_provenance(args.source)
    manifest = {
        "schema_version": CONTROL_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "command": sys.argv,
        "source": args.source,
        "split": args.split,
        "cases": len(rows),
        "data_root": str(args.data_root.resolve()),
        "dataset_cache_root": str(args.dataset_cache_root.resolve()),
        "input_root": (
            str(args.input_root.resolve()) if args.input_root is not None else None
        ),
        "output_root": str(args.output_root.resolve()),
        "taxonomy_sha256": sha256_file(
            args.dataset_cache_root / "taxonomy.json"
        ),
        "control_sizes": {
            "room": args.room_size,
            "structure": args.structure_size,
            "furniture": args.furniture_size,
            "support": args.support_size,
        },
        **provenance,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": args.device,
        },
        "rows": rows,
    }
    if rows and all("information" in row for row in rows):
        values = [row["information"] for row in rows]
        manifest["information_summary"] = {
            key: sum(item[key] for item in values) / len(values)
            for key in values[0]
        }
    return manifest


def main() -> None:
    args = parse_args()
    dataset = CubiCasaFloorplanDataset(
        args.data_root,
        args.split,
        image_size=256,
        cache_root=args.dataset_cache_root,
        limit=args.limit,
        augment=False,
        write_cache=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    if args.source == "oracle":
        rows = materialize_oracle(args, dataset, loader)
    else:
        rows = index_existing(args, dataset)
        measurements = measure_existing(args, dataset)
        for row in rows:
            row["information"] = measurements[row["name"]]
    validate_control_payload(
        read_control_cache(Path(rows[0]["file"])),
        dataset.taxonomy,
        room_size=args.room_size,
        structure_size=args.structure_size,
        furniture_size=args.furniture_size,
        support_size=args.support_size,
    )
    manifest = build_manifest(args, dataset, rows)
    path = args.output_root / args.split / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "manifest": str(path),
                "source": args.source,
                "split": args.split,
                "cases": len(rows),
                "information_summary": manifest.get("information_summary"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
