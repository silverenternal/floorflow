"""Evaluate the coarse-layout room refiner as a second-stage room branch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from floorflow.data.control_cache import CachedControlDataset
from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.data.retrieval_control_dataset import RetrievalControlDataset
from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.metrics import floorplan_metrics
from floorflow.evaluation.room_metric_suite import room_metric_suite
from floorflow.heads.coarse_layout_refiner import CoarseLayoutRefiner
from floorflow.heads.room_program import (
    gate_room_labels_to_program,
    room_presence_target,
)
from floorflow.inference.layered_decode import decode_layered_floorplan
from floorflow.representation.coarse_layout import coarse_room_layout_logits
from floorflow.training.checkpoint import load_checkpoint_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--refiner-checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path(".data/cubicasa5k"))
    parser.add_argument("--cache-root", type=Path, default=Path("outputs/cache/cubicasa5k_full_v2"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--layout-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--room-only", action="store_true")
    parser.add_argument("--control-cache", type=Path, default=None)
    parser.add_argument("--retrieval-controls", action="store_true")
    parser.add_argument(
        "--retrieval-metric",
        choices=("cosine", "mask_iou", "hybrid"),
        default="cosine",
    )
    parser.add_argument("--residual-layout", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted(set().union(*(row.keys() for row in rows)))
    result: dict[str, float] = {"case_count": float(len(rows))}
    for key in keys:
        values = np.asarray([row[key] for row in rows if key in row], dtype=np.float64)
        result[f"{key}_mean"] = float(values.mean())
        result[f"{key}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return result


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    taxonomy = FloorplanTaxonomy.read(args.cache_root / "taxonomy.json")
    base_model = None
    config = {"image_size": args.image_size}
    if not args.room_only:
        base_model, config, _taxonomy, _state = load_checkpoint_model(
            args.base_checkpoint,
            device,
            taxonomy=taxonomy,
        )
    refiner = CoarseLayoutRefiner(
        len(taxonomy.room_ids),
        residual=args.residual_layout,
    ).to(device)
    refiner_state = torch.load(args.refiner_checkpoint, map_location="cpu", weights_only=False)
    refiner.load_state_dict(refiner_state["model"])
    refiner.eval()
    dataset = CubiCasaFloorplanDataset(
        args.data_root,
        args.split,
        image_size=int(config.get("image_size", args.image_size)),
        cache_root=args.cache_root,
        limit=args.limit,
        augment=False,
    )
    if args.control_cache is not None:
        dataset = CachedControlDataset(
            dataset,
            args.control_cache,
            args.split,
        )
    elif args.retrieval_controls:
        dataset = RetrievalControlDataset(
            dataset,
            args.cache_root,
            args.split,
            metric=args.retrieval_metric,
            control_size=args.layout_size,
        )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    rows = []
    raster_dir = args.output_dir / "rasters"
    raster_dir.mkdir(parents=True, exist_ok=True)
    room_ids = torch.tensor(taxonomy.room_ids, device=device)
    for batch in loader:
        controls = batch[4] if len(batch) == 5 else None
        target, footprint, floor_ids, names = batch[:4]
        target = target.to(device)
        footprint = footprint.to(device)
        floor_ids = floor_ids.to(device)
        coarse = (
            controls["room_logits"].to(device)
            if controls is not None
            else coarse_room_layout_logits(target, taxonomy, size=args.layout_size)
        )
        refiner_logits = refiner(coarse, footprint)
        program = (
            controls["program"].to(device)
            if controls is not None
            else room_presence_target(target, taxonomy)
        )
        refiner_rooms = gate_room_labels_to_program(
            room_ids[refiner_logits.argmax(dim=1)],
            program,
            taxonomy.room_ids,
        ) * footprint.long()
        if args.room_only:
            prediction = refiner_rooms.long()
            layers = None
        else:
            layers = decode_layered_floorplan(
                base_model,
                target,
                footprint,
                taxonomy,
                room_labels=refiner_rooms,
                floor_ids=floor_ids,
            )
            prediction = layers.visible_labels()
        for index, name in enumerate(names):
            safe_name = str(name).replace("/", "__")
            target_room = torch.isin(target[index], room_ids)
            predicted_room = torch.isin(prediction[index], room_ids)
            metrics = floorplan_metrics(
                prediction[index][None],
                target[index][None],
                footprint[index][None],
                taxonomy,
                room_support=predicted_room[None],
            )
            suite = room_metric_suite(
                prediction[index].cpu().numpy(),
                target[index].cpu().numpy(),
                footprint[index].cpu().numpy(),
                taxonomy,
                predicted_room_support=predicted_room.cpu().numpy(),
                predicted_instance_support=predicted_room.cpu().numpy(),
                target_room_support=target_room.cpu().numpy(),
            )
            instance = suite["matched_instance"]
            metrics.update(
                {
                    "instance_geometric_iou": instance["matched_geometric_iou"],
                    "instance_room_f1_at_0_5": instance["instance_f1_at_threshold"],
                    "instance_type_accuracy": instance["matched_type_accuracy"],
                    "support_instance_geometric_iou": suite["support_instance"]["matched_geometric_iou"],
                    "support_instance_f1_at_0_5": suite["support_instance"]["instance_f1_at_threshold"],
                }
            )
            if layers is not None:
                layered_room = layers.room_labels
                layered_support = layers.room_support()
                layered_suite = room_metric_suite(
                    layered_room[index].cpu().numpy(),
                    target[index].cpu().numpy(),
                    footprint[index].cpu().numpy(),
                    taxonomy,
                    predicted_room_support=layered_support[index].cpu().numpy(),
                    predicted_instance_support=layered_support[index].cpu().numpy(),
                    target_room_support=target_room[index].cpu().numpy(),
                )
                layered_instance = layered_suite["matched_instance"]
                metrics.update(
                    {
                        "layered_room_macro_iou": layered_suite[
                            "visible_semantic"
                        ]["room_macro_iou"],
                        "layered_instance_geometric_iou": layered_instance[
                            "matched_geometric_iou"
                        ],
                        "layered_instance_room_f1_at_0_5": layered_instance[
                            "instance_f1_at_threshold"
                        ],
                        "layered_instance_type_accuracy": layered_instance[
                            "matched_type_accuracy"
                        ],
                    }
                )
            rows.append(metrics)
            np.savez_compressed(
                raster_dir / f"{safe_name}.npz",
                prediction=prediction[index].cpu().numpy(),
                target=target[index].cpu().numpy(),
                footprint=footprint[index].cpu().numpy(),
                room_support=predicted_room.cpu().numpy(),
                **(
                    {
                        "room_labels": layers.room_labels[index].cpu().numpy(),
                        "wall_layer": layers.wall[index].cpu().numpy(),
                        "opening_layer": layers.opening[index].cpu().numpy(),
                        "furniture_layer": layers.furniture[index].cpu().numpy(),
                    }
                    if layers is not None
                    else {}
                ),
            )
        print(f"evaluated {len(rows)}/{len(dataset)}", flush=True)
    metrics = aggregate(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (args.output_dir / "per_case.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
