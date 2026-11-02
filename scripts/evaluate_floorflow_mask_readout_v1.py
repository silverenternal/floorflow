"""Evaluate equivalent polygon and mask-partition FloorFlow readouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from floorflow.data.rplan_boundary_adapter import (
    RPlanBoundaryDataset,
    RPlanTensorLimits,
    collate_rplan_items,
    load_rplan_rows,
)
from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.canonical_floorplan import canonical_fingerprint
from floorflow.evaluation.room_metric_suite import matched_instance_metrics
from floorflow.model.unified_floorplan_generator import (
    UnifiedFloorplanGenerator,
    UnifiedGeneratorConfig,
    decode_generated_floorplan,
)
from floorflow.representation.unified_floorplan_tokens import (
    decode_floorplan,
    tokenize_floorplan,
)
from floorflow.representation.vector_floorplan_raster import (
    render_floorplan_rasters,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = (
    ROOT / ".data/external/CE2EPlan/dataset/dataset_json/data_val.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--val-offset", type=int, default=0)
    parser.add_argument("--val-cases", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument(
        "--mask-gap-policy",
        choices=(
            "threshold",
            "hole_fill",
            "nearest_fill",
            "partition_fill",
            "crack_fill",
            "bridge_fill",
            "adaptive_fill",
        ),
        default="threshold",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=("polygon", "mask_partition"),
        default=("polygon", "mask_partition"),
    )
    parser.add_argument(
        "--independent-masks",
        action="store_true",
        help="Ignore joint partition probabilities for diagnosis.",
    )
    parser.add_argument(
        "--partition-support",
        choices=("full_footprint", "independent_confidence"),
        default="independent_confidence",
    )
    parser.add_argument(
        "--wall-source",
        choices=(
            "room_polygons",
            "shared_owners",
            "shared_network",
            "shared_graph",
            "owner_cells",
        ),
        default="room_polygons",
    )
    parser.add_argument("--wall-fusion-weight", type=float, default=0.0)
    parser.add_argument("--wall-fusion-threshold", type=float, default=0.35)
    return parser.parse_args()


def _evaluate_plan(
    plan,
    target_labels: np.ndarray,
    taxonomy: FloorplanTaxonomy,
) -> dict[str, float | bool]:
    rendered = render_floorplan_rasters(plan, taxonomy, strict=False)
    labels = rendered["room_labels"]
    if target_labels.shape != labels.shape:
        target_labels = cv2.resize(
            np.asarray(target_labels),
            (labels.shape[1], labels.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    metrics = matched_instance_metrics(
        labels,
        target_labels,
        taxonomy,
        min_area=4,
    )
    encoded = tokenize_floorplan(plan, taxonomy)
    decoded = decode_floorplan(encoded.tokens, taxonomy)
    return {
        **{
            key: float(metrics[key])
            for key in (
                "matched_geometric_iou",
                "instance_f1_at_threshold",
                "matched_type_accuracy",
                "count_abs_error",
            )
        },
        "token_roundtrip_exact": (
            canonical_fingerprint(plan)
            == canonical_fingerprint(encoded.plan)
            == canonical_fingerprint(decoded)
        ),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and str(device) != "cuda:0":
        raise ValueError("RPLAN evaluation is restricted to cuda:0")
    payload = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    taxonomy = FloorplanTaxonomy.from_json(payload["taxonomy"])
    model = UnifiedFloorplanGenerator(
        taxonomy,
        UnifiedGeneratorConfig(**payload["config"]["model"]),
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    limits = RPlanTensorLimits(
        **{
            key: int(value)
            for key, value in payload["config"]["tensors"].items()
        }
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
    results = {
        source: {"rows": [], "failures": []}
        for source in args.sources
    }
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
                batch["entrance_points"],
                batch["entrance_valid"],
            )
            for index, case_name in enumerate(batch["case_name"]):
                single = {
                    key: value[index : index + 1]
                    for key, value in output.items()
                    if torch.is_tensor(value)
                }
                if args.independent_masks:
                    single.pop("room_instance_partition_probabilities", None)
                target = (
                    batch["target_labels"][index].detach().cpu().numpy()
                )
                for source in results:
                    try:
                        plan = decode_generated_floorplan(
                            single,
                            batch["boundary_points"][index : index + 1],
                            taxonomy,
                            case_id=case_name,
                            canvas_size=limits.canvas_size,
                            geometry_source=source,
                            confidence_threshold=float(
                                args.confidence_threshold
                            ),
                            mask_gap_policy=args.mask_gap_policy,
                            partition_support=args.partition_support,
                            wall_source=args.wall_source,
                            wall_fusion_weight=args.wall_fusion_weight,
                            wall_fusion_threshold=args.wall_fusion_threshold,
                            entrance_points=batch["entrance_points"][index],
                        )
                        results[source]["rows"].append(
                            _evaluate_plan(plan, target, taxonomy)
                        )
                    except (ValueError, RuntimeError) as error:
                        results[source]["failures"].append(
                            {"case": case_name, "error": str(error)}
                        )
    report = {
        "schema": "floorflow-mask-readout-evaluation-v1",
        "checkpoint": str(args.checkpoint),
        "validation_file": str(args.data),
        "validation_offset": int(args.val_offset),
        "requested_cases": int(args.val_cases),
        "test_opened": False,
        "device": str(device),
        "confidence_threshold": float(args.confidence_threshold),
        "mask_gap_policy": args.mask_gap_policy,
        "wall_source": args.wall_source,
        "wall_fusion_weight": args.wall_fusion_weight,
        "wall_fusion_threshold": args.wall_fusion_threshold,
        "sources": list(args.sources),
        "arms": {},
        "claim_boundary": (
            "Internal validation evidence; no RPLAN test or external ranking."
        ),
    }
    for source, result in results.items():
        rows_out = result["rows"]
        report["arms"][source] = {
            "successful_cases": len(rows_out),
            "failure_count": len(result["failures"]),
            "failures": result["failures"],
            "metrics": {
                key: float(np.mean([row[key] for row in rows_out]))
                for key in (
                    "matched_geometric_iou",
                    "instance_f1_at_threshold",
                    "matched_type_accuracy",
                    "count_abs_error",
                )
            },
            "token_roundtrip_exact_rate": float(
                np.mean([row["token_roundtrip_exact"] for row in rows_out])
            ),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                source: data["metrics"]
                for source, data in report["arms"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
