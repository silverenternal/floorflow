#!/usr/bin/env python3
"""Diagnose why furniture queries do or do not cross the render threshold."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_raster_predicted_control_bridge import RasterPredictedControlDataset
from train_residential_predicted_refiner import _predicted_controls

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.heads.dynamic_object_instance import (
    object_instance_targets,
    render_object_queries,
)
from floorflow.heads.room_mask_decoder import room_mask_metrics
from floorflow.inference.cad_stage_runtime import CadStageRuntime
from floorflow.model.predicted_channel_checkpoint import (
    build_predicted_channel_model,
)
from floorflow.training.cad_furniture_assignment import (
    room_assignment_targets,
)

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "outputs/train/cad_furniture_query_fast64_v1/best.pt",
    )
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/cad_furniture_query_fast64_diagnosis_v1.json",
    )
    return parser.parse_args()


def _f1(predicted: torch.Tensor, target: torch.Tensor) -> float:
    predicted = predicted.bool()
    target = target.bool()
    denominator = int(predicted.sum()) + int(target.sum())
    if denominator == 0:
        return 0.0
    return float(2.0 * (predicted & target).sum() / denominator)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    taxonomy = FloorplanTaxonomy.from_json(payload["taxonomy"])
    config = payload["config"]
    model = build_predicted_channel_model(taxonomy, config).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    stage_runtime = CadStageRuntime(
        taxonomy,
        device,
        room_checkpoint=config["room_checkpoint"],
        structure_checkpoint=config["structure_checkpoint"],
        centerline_checkpoint=config["centerline_checkpoint"],
        opening_checkpoint=config["opening_checkpoint"],
        room_blend=float(config["room_blend"]),
        structure_size=128,
    )
    dataset = RasterPredictedControlDataset(
        Path(config["predicted_root"]),
        "val",
        Path(config["cache_root"]),
        Path(config["data_root"]),
        image_size=int(config["image_size"]),
        limit=args.limit,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    threshold_rows = {
        threshold: []
        for threshold in (0.1, 0.2, 0.3, 0.4, 0.5)
    }
    rows = []
    query_mode = True
    target_furniture_ids = torch.tensor(
        taxonomy.furniture_ids,
        device=device,
    )
    for target, footprint, floor_ids, names, image in loader:
        target = target.to(device)
        footprint = footprint.to(device)
        controls = _predicted_controls(
            Path(config["predicted_root"]),
            "val",
            names,
            args,
            device,
        )
        controls, _ = stage_runtime(
            controls,
            footprint,
            image.to(device),
            floor_ids.to(device),
        )
        output = model(controls, footprint)
        if "furniture_query_class_logits" not in output:
            query_mode = False
            dense_probability = F.interpolate(
                output["cad_layers"].furniture_logits,
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).softmax(dim=1).amax(dim=1)
            classes, _masks, valid = object_instance_targets(
                target,
                taxonomy.furniture_ids,
                max_instances=24,
            )
            for threshold, threshold_values in threshold_rows.items():
                rendered = dense_probability >= threshold
                for index in range(target.shape[0]):
                    expected = torch.isin(
                        target[index],
                        target_furniture_ids,
                    ) & footprint[index].bool()
                    threshold_values.append(
                        _f1(rendered[index] & footprint[index].bool(), expected)
                    )
            for index in range(target.shape[0]):
                rows.append(
                    {
                        "query_budget": 0.0,
                        "presence_max": float(dense_probability[index].max()),
                        "presence_mean": float(dense_probability[index].mean()),
                        "mask_max": float(dense_probability[index].max()),
                        "mask_mean": float(dense_probability[index].mean()),
                        "presence_over_03": float(
                            (dense_probability[index] >= 0.3).float().mean()
                        ),
                        "mask_peak_over_03": float(
                            dense_probability[index].max() >= 0.3
                        ),
                        "target_instances": float(valid[index].sum()),
                    }
                )
            continue
        class_logits = output["furniture_query_class_logits"]
        mask_logits = output["furniture_query_mask_logits"]
        classes, masks, valid = object_instance_targets(
            target,
            taxonomy.furniture_ids,
            max_instances=int(class_logits.shape[1]),
        )
        resized_masks = F.interpolate(
            mask_logits,
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        metrics = room_mask_metrics(
            {
                "class_logits": class_logits,
                "mask_logits": resized_masks,
            },
            classes,
            masks,
            valid,
        )
        foreground = class_logits.softmax(dim=-1)[..., :-1]
        presence = foreground.amax(dim=-1)
        mask_probability = resized_masks.sigmoid()
        assignment_target = (
            room_assignment_targets(
                classes,
                masks,
                valid,
                target,
                taxonomy,
            )
            if "furniture_room_assignment_logits" in output
            else None
        )
        for threshold, threshold_values in threshold_rows.items():
            rendered = render_object_queries(
                class_logits,
                mask_logits,
                taxonomy.furniture_ids,
                footprint,
                presence_threshold=threshold,
                mask_threshold=threshold,
                active_queries=output["furniture_query_active"],
            )
            for index in range(target.shape[0]):
                expected = torch.isin(
                    target[index],
                    target_furniture_ids,
                ) & footprint[index].bool()
                threshold_values.append(
                    _f1(
                        (rendered[index] != 0) & footprint[index].bool(),
                        expected,
                    )
                )
        for index in range(target.shape[0]):
            active = output["furniture_query_active"][index]
            rows.append(
                {
                    **metrics,
                    "query_budget": float(
                        output["furniture_query_budget"][index].float()
                    ),
                    "presence_max": float(presence[index].max()),
                    "presence_mean": float(presence[index].mean()),
                    "mask_max": float(mask_probability[index].max()),
                    "mask_mean": float(mask_probability[index].mean()),
                    "presence_over_03": float(
                        (presence[index][active] >= 0.3).float().mean()
                    ),
                    "mask_peak_over_03": float(
                        (
                            mask_probability[index][active].amax(dim=(1, 2))
                            >= 0.3
                        ).float().mean()
                    ),
                    "target_instances": float(valid[index].sum()),
                    "room_assignment_accuracy": (
                        float(
                            (
                                output["furniture_room_assignment_logits"][
                                    index
                                ].argmax(dim=-1)
                                == assignment_target[index]
                            ).float().mean()
                        )
                        if assignment_target is not None
                        else 0.0
                    ),
                }
            )
    summary_keys = tuple(rows[0])
    report = {
        "schema_version": "cad-furniture-query-diagnosis-v1",
        "checkpoint": str(args.checkpoint),
        "cases": len(rows),
        "mode": "furniture_query" if query_mode else "dense_furniture",
        "test_opened": False,
        "summary": {
            key: float(np.mean([row[key] for row in rows]))
            for key in summary_keys
        },
        "threshold_sweep": {
            str(threshold): float(np.mean(values))
            for threshold, values in threshold_rows.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
