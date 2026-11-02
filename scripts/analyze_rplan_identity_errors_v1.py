"""Decompose RPLAN room identity, count, and routing errors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from floorflow.data.rplan_boundary_adapter import (
    RPlanBoundaryDataset,
    RPlanTensorLimits,
    collate_rplan_items,
    load_rplan_rows,
)
from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.heads.polygon_loop import room_query_metrics
from floorflow.heads.room_mask_decoder import room_mask_metrics
from floorflow.model.unified_floorplan_generator import (
    UnifiedFloorplanGenerator,
    UnifiedGeneratorConfig,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = (
    ROOT / ".data/external/CE2EPlan/dataset/dataset_json/data_val.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--val-offset", type=int, default=0)
    parser.add_argument("--val-cases", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _mask_metrics(
    output: dict[str, torch.Tensor],
    batch: dict,
    taxonomy: FloorplanTaxonomy,
) -> dict[str, float]:
    mask_logits = output["room_instance_mask_logits"]
    size = mask_logits.shape[-2:]
    target_masks = F.interpolate(
        batch["room_masks"].flatten(0, 1)[:, None],
        size=size,
        mode="bilinear",
        align_corners=False,
    ).reshape(
        batch["room_masks"].shape[0],
        batch["room_masks"].shape[1],
        *size,
    )
    target_labels = F.interpolate(
        batch["target_labels"][:, None].float(),
        size=size,
        mode="nearest",
    )[:, 0].long()
    footprint = F.interpolate(
        batch["boundary_mask"].float(),
        size=size,
        mode="bilinear",
        align_corners=False,
    )[:, 0]
    return room_mask_metrics(
        {
            "class_logits": output["room_instance_class_logits"],
            "mask_logits": mask_logits,
        },
        batch["room_classes"],
        target_masks,
        batch["room_exists"],
        taxonomy,
        target_labels=target_labels,
        footprint=footprint,
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and str(device) != "cuda:0":
        raise ValueError("RPLAN diagnostics are restricted to cuda:0")
    payload = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    config = payload["config"]
    taxonomy = FloorplanTaxonomy.from_json(payload["taxonomy"])
    model_config = dict(config["model"])
    if any(
        key.startswith("query_quality.")
        for key in payload["model"]
    ):
        model_config["enable_query_quality"] = True
    model = UnifiedFloorplanGenerator(
        taxonomy,
        UnifiedGeneratorConfig(**model_config),
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    limits = RPlanTensorLimits(
        **{key: int(value) for key, value in config["tensors"].items()}
    )
    rows = load_rplan_rows(
        args.data,
        offset=int(args.val_offset),
        limit=int(args.val_cases),
    )
    dataset = RPlanBoundaryDataset(
        args.data,
        taxonomy,
        rows=rows,
        limits=limits,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.workers),
        collate_fn=collate_rplan_items,
        pin_memory=device.type == "cuda",
    )
    totals = {}
    count_confusion = np.zeros((13, 13), dtype=np.int64)
    signed_count_errors = []
    seen = 0
    with torch.inference_mode():
        for raw_batch in loader:
            batch = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in raw_batch.items()
            }
            output = model(
                batch["boundary_mask"],
                batch["boundary_valid"],
                batch["boundary_points"],
            )
            predicted_count = output["room_count_logits"].argmax(dim=-1)
            target_count = batch["room_count"]
            for predicted, target in zip(
                predicted_count.cpu().tolist(),
                target_count.cpu().tolist(),
                strict=True,
            ):
                count_confusion[int(target), int(predicted)] += 1
            signed_count_errors.extend(
                (predicted_count - target_count).cpu().tolist()
            )
            query = room_query_metrics(
                {
                    "class_logits": output["room_class"],
                    "polygons": output["room_polygons"],
                    "valid_logits": output["room_exists"],
                },
                batch["target_labels"],
                taxonomy,
                validity_threshold=0.25,
                min_area=4,
                query_budget=predicted_count,
                query_quality_logits=output.get("query_quality_logits"),
            )
            mask = _mask_metrics(output, batch, taxonomy)
            batch_size = batch["boundary_mask"].shape[0]
            for key, value in {**query, **mask}.items():
                totals[key] = totals.get(key, 0.0) + float(value) * batch_size
            routing = output["routing"]
            for key in (
                "routing_entropy",
                "attention_entropy",
                "effective_expert_fraction",
                "top_k",
            ):
                totals[f"routing_{key}"] = totals.get(
                    f"routing_{key}",
                    0.0,
                ) + float(routing[key]) * batch_size
            seen += batch_size
    metrics = {key: value / max(1, seen) for key, value in totals.items()}
    errors = np.asarray(signed_count_errors, dtype=np.int64)
    report = {
        "schema": "rplan-identity-error-decomposition-v1",
        "checkpoint": str(args.checkpoint),
        "validation_file": str(args.data),
        "validation_offset": int(args.val_offset),
        "cases": seen,
        "test_opened": False,
        "device": str(device),
        "metrics": metrics,
        "count_error": {
            "mean": float(errors.mean()) if errors.size else 0.0,
            "mean_absolute": float(np.abs(errors).mean()) if errors.size else 0.0,
            "under_count_rate": float((errors < 0).mean()) if errors.size else 0.0,
            "exact_count_rate": float((errors == 0).mean()) if errors.size else 0.0,
            "over_count_rate": float((errors > 0).mean()) if errors.size else 0.0,
            "signed_histogram": {
                str(value): int((errors == value).sum())
                for value in range(-8, 9)
            },
        },
        "count_confusion": {
            "target_rows": list(range(13)),
            "predicted_columns": list(range(13)),
            "matrix": count_confusion.tolist(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"cases": seen, **metrics}, indent=2))


if __name__ == "__main__":
    main()
