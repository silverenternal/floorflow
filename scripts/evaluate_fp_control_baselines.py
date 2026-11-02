#!/usr/bin/env python3
"""Evaluate matched layer baselines reconstructed from serialized controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from floorflow.data.control_cache import CachedControlDataset
from floorflow.data.control_contract import CONTROL_SCHEMA_VERSION, sha256_file
from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.evaluation.completion_utility import completion_utility_metrics
from floorflow.evaluation.control_baseline import control_layers
from floorflow.evaluation.fpeval_bridge import fpeval_raster_metrics
from floorflow.evaluation.room_metric_suite import room_metric_suite
from floorflow.heads.structured_objects import (
    object_group_metrics,
    partition_wall_mask,
)
from floorflow.representation.layered_floorplan import LayeredFloorplan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path(".data/cubicasa5k"))
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("outputs/cache/cubicasa5k_full_v2"),
    )
    parser.add_argument(
        "--geometry-root",
        type=Path,
        default=Path("outputs/cache/cubicasa5k_geometry_v1"),
    )
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--method", choices=("raw_control",), default="raw_control")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_dataset(
    args: argparse.Namespace,
) -> tuple[CachedControlDataset, dict[str, object]]:
    base = CubiCasaFloorplanDataset(
        args.data_root,
        args.split,
        image_size=args.image_size,
        cache_root=args.cache_root,
        limit=args.limit,
        augment=False,
        write_cache=False,
    )
    manifest_path = args.control_root / args.split / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["schema_version"] != CONTROL_SCHEMA_VERSION:
        raise ValueError("control schema mismatch")
    if manifest["split"] != args.split:
        raise ValueError("control split mismatch")
    if args.limit is None and int(manifest["cases"]) != len(base):
        raise ValueError("control manifest case count mismatch")
    if args.limit is not None and len(base) > int(manifest["cases"]):
        raise ValueError("dataset limit exceeds control manifest")
    manifest["manifest_path"] = str(manifest_path)
    manifest["manifest_sha256"] = sha256_file(manifest_path)
    return CachedControlDataset(base, args.control_root, args.split), manifest


def support_iou(prediction, target) -> float:
    union = np.logical_or(prediction, target).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(prediction, target).sum() / union)


def geometry_path(root: Path, split: str, name: str) -> Path:
    floor = Path(name.replace("__", "/"))
    return (
        root
        / split
        / floor.parent
        / f"{floor.name.replace('-', '_')}.geometry.npz"
    )


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dataset, manifest = load_dataset(args)
    taxonomy = dataset.taxonomy
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    room_ids = torch.tensor(taxonomy.room_ids, device=device)
    rows: list[dict[str, float]] = []
    raster_dir = args.output_dir / "rasters"
    raster_dir.mkdir(parents=True, exist_ok=True)
    for target, footprint, _floor_ids, names, controls in loader:
        target = target.to(device)
        footprint = footprint.to(device)
        controls = {key: value.to(device) for key, value in controls.items()}
        predictions = []
        topology_predictions = []
        object_parts = []
        for index in range(target.shape[0]):
            rooms, wall, openings, furniture = control_layers(
                {key: value[index] for key, value in controls.items()},
                footprint[index],
                taxonomy,
            )
            layer = LayeredFloorplan(
                room_labels=rooms[None],
                wall=wall[None],
                opening=openings[None],
                furniture=furniture[None],
                footprint=footprint[index : index + 1],
                taxonomy=taxonomy,
            )
            prediction = layer.visible_labels()[0]
            topology_wall = wall | partition_wall_mask(rooms[None], radius=2)[0]
            topology_wall = F.max_pool2d(
                topology_wall.float()[None, None],
                kernel_size=3,
                stride=1,
                padding=1,
            )[0, 0] > 0.5
            closed = LayeredFloorplan(
                room_labels=rooms[None],
                wall=topology_wall.long()[None],
                opening=openings[None],
                furniture=furniture[None],
                footprint=footprint[index : index + 1],
                taxonomy=taxonomy,
            ).visible_labels()[0]
            predictions.append(prediction)
            topology_predictions.append(closed)
            object_parts.append((wall, openings, furniture))
        prediction = torch.stack(predictions)
        closed_prediction = torch.stack(topology_predictions)

        for index, name in enumerate(names):
            room_support = torch.isin(
                prediction[index],
                room_ids,
            ).cpu().numpy()
            target_room_support = torch.isin(
                target[index],
                room_ids,
            ).cpu().numpy()
            suite = room_metric_suite(
                prediction[index].cpu().numpy(),
                target[index].cpu().numpy(),
                footprint[index].cpu().numpy(),
                taxonomy,
                predicted_room_support=room_support,
                predicted_instance_support=room_support,
                target_room_support=target_room_support,
            )
            channel = object_group_metrics(
                {
                    "wall": object_parts[index][0][None],
                    "opening": object_parts[index][1][None],
                    "furniture": object_parts[index][2][None],
                },
                target[index : index + 1],
                taxonomy,
            )
            fpeval = fpeval_raster_metrics(
                prediction[index].cpu().numpy(),
                target[index].cpu().numpy(),
                taxonomy,
                tolerance_fractions=(0.05,),
            )
            closed_fpeval = fpeval_raster_metrics(
                closed_prediction[index].cpu().numpy(),
                target[index].cpu().numpy(),
                taxonomy,
                tolerance_fractions=(0.05,),
            )
            room_fpeval = fpeval_raster_metrics(
                prediction[index].cpu().numpy(),
                target[index].cpu().numpy(),
                taxonomy,
                tolerance_fractions=(0.05,),
                explicit_prediction_rooms=True,
                explicit_target_rooms=True,
            )
            path = geometry_path(args.geometry_root, args.split, name)
            with np.load(path) as geometry:
                geometry_support = geometry["room_support"] > 0
            summary = fpeval["summary"]["0.05"]
            rows.append(
                {
                    "room_macro_iou": suite["visible_semantic"]["room_macro_iou"],
                    "instance_geometric_iou": suite["matched_instance"][
                        "matched_geometric_iou"
                    ],
                    "instance_room_f1": suite["matched_instance"][
                        "instance_f1_at_threshold"
                    ],
                    "instance_type_accuracy": suite["matched_instance"][
                        "matched_type_accuracy"
                    ],
                    **channel,
                    "fpeval_wall_f1": summary["wall_f1"],
                    "fpeval_opening_f1": summary["opening_f1"],
                    "fpeval_room_polygon_f1": room_fpeval["summary"]["0.05"][
                        "room_f1"
                    ],
                    "fpeval_room_face_f1": summary["room_f1"],
                    "fpeval_validity": float(
                        fpeval["prediction_validity"]["valid"]
                    ),
                    "fpeval_watertight": float(
                        closed_fpeval["prediction_validity"]["watertight"]
                    ),
                    "target_watertight": float(
                        closed_fpeval["target_validity"]["watertight"]
                    ),
                    "room_support_iou": support_iou(
                        room_support,
                        geometry_support,
                    ),
                    "room_support_iou_visible": support_iou(
                        room_support,
                        geometry_support,
                    ),
                    **completion_utility_metrics(
                        prediction[index].cpu().numpy(),
                        prediction[index].cpu().numpy(),
                        target[index].cpu().numpy(),
                        taxonomy,
                        output_edit_cost=summary["edit_cost"],
                        output_edit_cost_per_gt_wall=summary[
                            "edit_cost_per_gt_wall"
                        ],
                    ),
                }
            )
            np.savez_compressed(
                raster_dir / f"{name.replace('/', '__')}.npz",
                prediction=prediction[index].cpu().numpy().astype(np.uint8),
                target=target[index].cpu().numpy().astype(np.uint8),
                footprint=footprint[index].cpu().numpy().astype(np.uint8),
            )
        print(f"evaluated {len(rows)}/{len(dataset)}", flush=True)

    metrics = {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]
    }
    target_watertight = [row["target_watertight"] > 0.5 for row in rows]
    metrics["conditional_watertight"] = float(
        np.mean(
            [
                row["fpeval_watertight"]
                for row, valid in zip(rows, target_watertight)
                if valid
            ]
        )
    )
    report = {
        "cases": len(rows),
        "method": args.method,
        "task_variant": "serialized control reconstruction baseline",
        "best_of_n_used": False,
        "single_deterministic_sample": True,
        "control_contract": {
            "source": manifest["source"],
            "target_used_for_control": manifest["target_used_for_control"],
            "deployable": manifest["deployable"],
            "root": manifest["output_root"],
            "manifest": manifest["manifest_path"],
            "manifest_sha256": manifest["manifest_sha256"],
            "taxonomy_sha256": manifest["taxonomy_sha256"],
        },
        "metrics": metrics,
        "target_watertight_fraction": float(np.mean(target_watertight)),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "per_case.json").write_text(
        json.dumps(rows, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
