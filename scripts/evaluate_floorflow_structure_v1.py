"""Evaluate canonical structural legality and graph metrics for FloorFlow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from floorflow.data.rplan_boundary_adapter import (
    RPlanBoundaryDataset,
    RPlanTensorLimits,
    collate_rplan_items,
    load_rplan_rows,
    rplan_row_to_canonical,
)
from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.fevp_metric_contracts import FEVPTolerances
from floorflow.evaluation.sota_protocol import evaluate_canonical_pair
from floorflow.heads.cad_wall_candidate_scorer import CadWallCandidateScorer
from floorflow.inference.owner_wall_action import (
    build_owner_wall_action_batch,
)
from floorflow.model.cad_action_graph import STAGES
from floorflow.model.unified_floorplan_generator import (
    UnifiedFloorplanGenerator,
    UnifiedGeneratorConfig,
    decode_generated_floorplan,
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
    parser.add_argument(
        "--case-indices",
        default="",
        help="Optional comma-separated zero-based validation row indices.",
    )
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
    parser.add_argument("--ged-timeout", type=float, default=1.0)
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
    parser.add_argument("--owner-wall-action-checkpoint", type=Path)
    parser.add_argument("--owner-wall-action-threshold", type=float, default=0.4)
    parser.add_argument(
        "--per-case",
        action="store_true",
        help="Include per-case structure rows in the report.",
    )
    return parser.parse_args()


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
    wall_action = None
    action_max_walls = 0
    if args.owner_wall_action_checkpoint is not None:
        action_payload = torch.load(
            args.owner_wall_action_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        action_config = action_payload["config"]
        action_max_walls = int(action_config["max_walls"])
        wall_action = CadWallCandidateScorer(
            in_channels=1,
            hidden_dim=int(action_config["hidden_dim"]),
            graph_dim=int(action_config["hidden_dim"]),
            residual_scale=0.01,
            use_encoded_score=bool(
                action_config.get("use_encoded_score", False)
            ),
        ).to(device)
        wall_action.load_state_dict(action_payload["scorer"])
        wall_action.eval()
        action_projection = torch.nn.Linear(
            int(action_config["hidden_dim"]),
            int(action_config["hidden_dim"]),
        ).to(device)
        action_projection.load_state_dict(
            action_payload["graph_projection"]
        )
        action_projection.eval()
    limits = RPlanTensorLimits(
        **{
            key: int(value)
            for key, value in payload["config"]["tensors"].items()
        }
    )
    if args.case_indices:
        indices = [int(value) for value in args.case_indices.split(",")]
        available = load_rplan_rows(
            args.data,
            offset=0,
            limit=max(indices) + 1,
        )
        raw_rows = [available[index] for index in indices]
    else:
        raw_rows = load_rplan_rows(
            args.data,
            offset=int(args.val_offset),
            limit=int(args.val_cases),
        )
    dataset = RPlanBoundaryDataset(
        args.data,
        taxonomy,
        rows=raw_rows,
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
    tolerances = FEVPTolerances()
    metrics = []
    failures = []
    cursor = 0
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
            action_outputs = None
            action_candidates = None
            if wall_action is not None:
                action_candidates = build_owner_wall_action_batch(
                    output,
                    batch["boundary_points"],
                    taxonomy,
                    canvas_size=limits.canvas_size,
                    max_walls=action_max_walls,
                )
                stage_tokens = torch.stack(
                    [
                        output["action_tokens"][name]
                        for name in STAGES
                    ],
                    dim=1,
                )
                action_outputs = wall_action(
                    batch["boundary_mask"].float(),
                    action_candidates.endpoints,
                    action_candidates.valid,
                    action_candidates.evidence,
                    action_projection(stage_tokens),
                    action_candidates.exterior,
                )
            for index, case_name in enumerate(batch["case_name"]):
                single = {
                    key: value[index : index + 1]
                    for key, value in output.items()
                    if torch.is_tensor(value)
                }
                if args.independent_masks:
                    single.pop("room_instance_partition_probabilities", None)
                try:
                    prediction = decode_generated_floorplan(
                        single,
                        batch["boundary_points"][index : index + 1],
                        taxonomy,
                        case_id=case_name,
                        canvas_size=limits.canvas_size,
                        confidence_threshold=float(
                            args.confidence_threshold
                        ),
                        geometry_source="mask_partition",
                        mask_gap_policy=args.mask_gap_policy,
                        partition_support=args.partition_support,
                        wall_source=args.wall_source,
                        wall_fusion_weight=args.wall_fusion_weight,
                        wall_fusion_threshold=args.wall_fusion_threshold,
                        wall_action_endpoints=(
                            action_candidates.endpoints[index]
                            if action_candidates is not None
                            else None
                        ),
                        wall_action_scores=(
                            action_outputs["score_logits"][index].sigmoid()
                            if action_outputs is not None
                            else None
                        ),
                        wall_action_valid=(
                            action_candidates.valid[index]
                            if action_candidates is not None
                            else None
                        ),
                        wall_action_exterior=(
                            action_candidates.exterior[index]
                            if action_candidates is not None
                            else None
                        ),
                        wall_action_threshold=(
                            args.owner_wall_action_threshold
                        ),
                        entrance_points=batch["entrance_points"][index],
                    )
                    target = rplan_row_to_canonical(
                        raw_rows[cursor],
                        taxonomy,
                        image_size=limits.canvas_size,
                    )
                    row = evaluate_canonical_pair(
                        prediction,
                        target,
                        taxonomy,
                        tolerances=tolerances,
                        ged_timeout_seconds=float(args.ged_timeout),
                    )
                    closure = next(
                        (
                            operation
                            for operation in prediction.operations
                            if (
                                operation.kind.endswith("wall_projection")
                                or operation.kind
                                == "owner_action_wall_selection"
                            )
                        ),
                        None,
                    )
                    connector_count = 0
                    if closure is not None:
                        marker = "connectors="
                        if marker in closure.detail:
                            connector_count = int(
                                closure.detail.split(marker, 1)[1]
                            )
                    row["wall_count"] = float(len(prediction.walls))
                    row["wall_connector_count"] = float(connector_count)
                    connector_walls = [
                        wall
                        for wall in prediction.walls
                        if wall.id.startswith("connector_")
                    ]
                    connector_lengths = [
                        float(
                            np.linalg.norm(
                                np.subtract(wall.end, wall.start)
                            )
                        )
                        for wall in connector_walls
                    ]
                    all_wall_length = sum(
                        float(
                            np.linalg.norm(
                                np.subtract(wall.end, wall.start)
                            )
                        )
                        for wall in prediction.walls
                    )
                    row["wall_connector_length_mean"] = (
                        float(np.mean(connector_lengths))
                        if connector_lengths
                        else 0.0
                    )
                    row["wall_connector_length_max"] = (
                        float(np.max(connector_lengths))
                        if connector_lengths
                        else 0.0
                    )
                    row["wall_connector_length_ratio"] = (
                        float(sum(connector_lengths) / all_wall_length)
                        if all_wall_length
                        else 0.0
                    )
                    stage_exits = [
                        operation
                        for operation in prediction.operations
                        if operation.kind == "cad_stage_exit"
                    ]
                    row["cad_stage_exit_count"] = float(len(stage_exits))
                    row["cad_stage_incomplete_count"] = float(
                        sum(
                            "status=incomplete" in operation.detail
                            for operation in stage_exits
                        )
                    )
                    metrics.append(row)
                except (ValueError, RuntimeError) as error:
                    failures.append(
                        {"case": case_name, "error": str(error)}
                    )
                cursor += 1
    numeric_keys = sorted(
        {
            key
            for row in metrics
            for key, value in row.items()
            if isinstance(value, (int, float)) and key != "case_id"
        }
    )
    summary = {
        key: float(np.mean([float(row[key]) for row in metrics]))
        for key in numeric_keys
        if metrics
    }
    bool_keys = sorted(
        {
            key
            for row in metrics
            for key, value in row.items()
            if isinstance(value, bool)
        }
    )
    for key in bool_keys:
        summary[key] = float(np.mean([bool(row[key]) for row in metrics]))
    report = {
        "schema": "floorflow-canonical-structure-v1",
        "checkpoint": str(args.checkpoint),
        "validation_file": str(args.data),
        "validation_offset": -1 if args.case_indices else int(args.val_offset),
        "case_indices": [
            int(value) for value in args.case_indices.split(",") if value
        ],
        "requested_cases": len(raw_rows),
        "successful_cases": len(metrics),
        "failure_count": len(failures),
        "failures": failures,
        "test_opened": False,
        "confidence_threshold": float(args.confidence_threshold),
        "mask_gap_policy": args.mask_gap_policy,
        "wall_source": args.wall_source,
        "wall_fusion_weight": args.wall_fusion_weight,
        "wall_fusion_threshold": args.wall_fusion_threshold,
        "owner_wall_action_checkpoint": (
            str(args.owner_wall_action_checkpoint)
            if args.owner_wall_action_checkpoint is not None
            else None
        ),
        "owner_wall_action_threshold": args.owner_wall_action_threshold,
        "ged_timeout_seconds": float(args.ged_timeout),
        "metrics": summary,
        **({"rows": metrics} if args.per_case else {}),
        "claim_boundary": (
            "Internal RPLAN validation structure evidence; not a test or "
            "cross-paper leaderboard."
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
                "successful_cases": len(metrics),
                "failures": len(failures),
                "metrics": summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
