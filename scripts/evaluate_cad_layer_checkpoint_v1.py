#!/usr/bin/env python3
"""Evaluate a trained CAD layered checkpoint with graph and safety metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from shapely.geometry import LineString, Point
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_raster_predicted_control_bridge import RasterPredictedControlDataset
from train_residential_predicted_refiner import _predicted_controls

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.fpeval_bridge import evaluate, to_fpeval_plan, verify
from floorflow.evaluation.furniture_vector_metrics import furniture_vector_metrics
from floorflow.evaluation.generation_plausibility import generation_plausibility
from floorflow.evaluation.housemind_structure import housemind_style_metrics
from floorflow.evaluation.room_metric_suite import room_metric_suite
from floorflow.evaluation.structural_legality import structural_legality_metrics
from floorflow.geometry.wall_graph_readout import raster_labels_to_wall_plan
from floorflow.heads.dynamic_object_instance import object_instance_targets
from floorflow.inference.cad_layer_vectorizer import vectorize_layers
from floorflow.inference.cad_stage_runtime import CadStageRuntime
from floorflow.model.cad_layers import CadLayeredOutput
from floorflow.model.predicted_channel_checkpoint import (
    build_predicted_channel_model,
)
from floorflow.training.typed_channel_eval import (
    compose_stage_labels,
    family_metrics,
)

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wall-threshold", type=float, default=0.5)
    parser.add_argument("--opening-threshold", type=float, default=0.5)
    parser.add_argument("--furniture-threshold", type=float, default=0.5)
    parser.add_argument("--minimum-room-area", type=int, default=16)
    parser.add_argument("--largest-room-component-only", action="store_true")
    parser.add_argument("--vector-layer-readout", action="store_true")
    parser.add_argument("--room-boundary-candidates", action="store_true")
    parser.add_argument("--candidate-evidence-threshold", type=float, default=0.2)
    parser.add_argument("--reconcile-wall-candidates", action="store_true")
    parser.add_argument("--runtime-wall-probability", action="store_true")
    parser.add_argument("--use-junction-stage", action="store_true")
    parser.add_argument("--vector-plan-output-dir", type=Path, default=None)
    parser.add_argument("--skip-generation-plausibility", action="store_true")
    parser.add_argument("--room-program-bias", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _jsonable(value):
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _wall_length(walls) -> float:
    return sum(
        float(np.linalg.norm(np.subtract(wall.end, wall.start)))
        for wall in walls
    )


def _wall_endpoint_contact_rate(walls) -> float:
    if not walls:
        return 0.0
    lines = [
        LineString([wall.start, wall.end])
        for wall in walls
    ]
    contacts = 0
    endpoints = 0
    for wall_index, wall in enumerate(walls):
        for endpoint in (wall.start, wall.end):
            endpoints += 1
            if any(
                line.distance(Point(endpoint)) <= 2.0
                for index, line in enumerate(lines)
                if index != wall_index
            ):
                contacts += 1
    return contacts / max(1, endpoints)


def _opening_match_errors(predicted, target) -> dict[str, float]:
    if not predicted or not target:
        return {
            "opening_center_error_proxy": 0.0,
            "opening_width_error_proxy": 0.0,
        }
    cost = np.asarray(
        [
            [
                np.linalg.norm(
                    np.subtract(left.center, right.center)
                )
                for right in target
            ]
            for left in predicted
        ],
        dtype=np.float64,
    )
    rows, columns = linear_sum_assignment(cost)
    return {
        "opening_center_error_proxy": float(cost[rows, columns].mean()),
        "opening_width_error_proxy": float(
            np.mean(
                [
                    abs(predicted[row].width - target[column].width)
                    for row, column in zip(rows, columns, strict=True)
                ]
            )
        ),
    }


def _furniture_proxy_metrics(furniture) -> dict[str, float]:
    count = len(furniture)
    return {
        "furniture_attachment_rate_proxy": (
            sum(item.room_index >= 0 for item in furniture) / max(1, count)
        ),
        "furniture_polygon_validity_proxy": (
            sum(len(item.polygon) >= 4 for item in furniture)
            / max(1, count)
        ),
        "furniture_confidence_mean_proxy": (
            float(np.mean([item.confidence for item in furniture]))
            if furniture
            else 0.0
        ),
    }


@torch.inference_mode()
def main() -> None:
    cli = parse_args()
    device = torch.device(cli.device)
    payload = torch.load(cli.checkpoint, map_location="cpu", weights_only=False)
    config = payload["config"]
    taxonomy = FloorplanTaxonomy.from_json(payload["taxonomy"])
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
        use_junction_stage=cli.use_junction_stage,
        room_program_bias=cli.room_program_bias,
    )
    dataset = RasterPredictedControlDataset(
        Path(config["predicted_root"]),
        "val",
        Path(config["cache_root"]),
        Path(config["data_root"]),
        image_size=int(config["image_size"]),
        limit=cli.limit,
    )
    loader = DataLoader(
        dataset,
        batch_size=cli.batch_size,
        shuffle=False,
        num_workers=cli.workers,
        pin_memory=True,
    )
    rows = []
    room_ids = torch.tensor(taxonomy.room_ids, device=device)

    class ControlArgs:
        predicted_root = Path(config["predicted_root"])
        room_size = 224
        structure_size = 128

    for target, footprint, floor_ids, names, image in loader:
        target = target.to(device)
        footprint = footprint.to(device)
        controls = _predicted_controls(
            Path(config["predicted_root"]),
            "val",
            names,
            ControlArgs(),
            device,
        )
        controls, _diagnostics = stage_runtime(
            controls,
            footprint,
            image.to(device),
            floor_ids.to(device),
        )
        opening_family_probability = torch.nn.functional.interpolate(
            controls["opening_family_probability"].float(),
            size=tuple(target.shape[-2:]),
            mode="bilinear",
            align_corners=False,
        )
        room_count = len(taxonomy.room_ids)
        runtime_wall_probability = torch.nn.functional.interpolate(
            controls["structure"][:, room_count : room_count + 1].float(),
            size=tuple(target.shape[-2:]),
            mode="bilinear",
            align_corners=False,
        )
        output = model(controls, footprint)
        rendered, visible = compose_stage_labels(
            output,
            tuple(target.shape[-2:]),
            footprint,
            taxonomy,
            wall_threshold=cli.wall_threshold,
            opening_threshold=cli.opening_threshold,
            furniture_threshold=cli.furniture_threshold,
        )
        for index in range(target.shape[0]):
            room_prediction = rendered.room_labels[index]
            room_support = torch.isin(room_prediction, room_ids)
            target_support = torch.isin(target[index], room_ids)
            suite = room_metric_suite(
                room_prediction.cpu().numpy(),
                target[index].cpu().numpy(),
                footprint[index].cpu().numpy(),
                taxonomy,
                predicted_room_support=room_support.cpu().numpy(),
                predicted_instance_support=room_support.cpu().numpy(),
                target_room_support=target_support.cpu().numpy(),
            )
            visible_np = visible[index].cpu().numpy()
            target_np = target[index].cpu().numpy()
            footprint_np = footprint[index].cpu().numpy()
            target_plan = raster_labels_to_wall_plan(target_np, taxonomy)
            graph = housemind_style_metrics(
                visible_np,
                target_np,
                footprint_np,
                taxonomy,
                ged_timeout_seconds=0.5,
            )
            structural = structural_legality_metrics(
                visible_np,
                target_np,
                taxonomy,
            )
            plausibility = (
                {}
                if cli.skip_generation_plausibility
                else generation_plausibility(
                    visible_np,
                    footprint_np,
                    taxonomy,
                    explicit_rooms=True,
                )
            )
            layer = family_metrics(
                type(rendered)(
                    room_labels=rendered.room_labels[index : index + 1],
                    wall=rendered.wall[index : index + 1],
                    opening=rendered.opening[index : index + 1],
                    furniture=rendered.furniture[index : index + 1],
                    footprint=rendered.footprint[index : index + 1],
                    taxonomy=taxonomy,
                ),
                target[index],
                footprint[index],
                taxonomy,
            )
            vector_metrics = {}
            if cli.vector_layer_readout:
                resized_layers = output["cad_layers"].resized(
                    tuple(target.shape[-2:])
                )
                furniture_queries = None
                if "furniture_query_class_logits" in output:
                    furniture_queries = {
                        "class_logits": output[
                            "furniture_query_class_logits"
                        ][index : index + 1],
                        "mask_logits": output[
                            "furniture_query_mask_logits"
                        ][index : index + 1],
                        "query_budget": output[
                            "furniture_query_budget"
                        ][index : index + 1],
                        "active_queries": output[
                            "furniture_query_active"
                        ][index : index + 1],
                    }
                    if "furniture_room_assignment_logits" in output:
                        furniture_queries["room_assignment_logits"] = output[
                            "furniture_room_assignment_logits"
                        ][index : index + 1]
                vector_metrics_raw = vectorize_layers(
                    CadLayeredOutput(
                        room_logits=resized_layers.room_logits[index : index + 1],
                        wall_logits=resized_layers.wall_logits[index : index + 1],
                        opening_logits=resized_layers.opening_logits[
                            index : index + 1
                        ],
                        furniture_logits=resized_layers.furniture_logits[
                            index : index + 1
                        ],
                        support_logits=(
                            resized_layers.support_logits[index : index + 1]
                            if resized_layers.support_logits is not None
                            else None
                        ),
                    ),
                    footprint[index : index + 1],
                    taxonomy,
                    wall_threshold=cli.wall_threshold,
                    opening_threshold=cli.opening_threshold,
                    furniture_threshold=cli.furniture_threshold,
                    minimum_room_area=cli.minimum_room_area,
                    use_room_boundary_candidates=cli.room_boundary_candidates,
                    candidate_evidence_threshold=cli.candidate_evidence_threshold,
                    reconcile_wall_candidates=cli.reconcile_wall_candidates,
                    opening_family_probability=(
                        opening_family_probability[index].cpu().numpy()
                    ),
                    wall_probability=(
                        runtime_wall_probability[index, 0].cpu().numpy()
                        if cli.runtime_wall_probability
                        else None
                    ),
                    opening_wall_probability=(
                        runtime_wall_probability[index, 0].cpu().numpy()
                    ),
                    largest_room_component_only=cli.largest_room_component_only,
                    furniture_queries=furniture_queries,
                )
                if cli.vector_plan_output_dir is not None:
                    cli.vector_plan_output_dir.mkdir(
                        parents=True,
                        exist_ok=True,
                    )
                    plan_path = (
                        cli.vector_plan_output_dir
                        / f"{str(names[index]).replace('/', '_')}.json"
                    )
                    plan_path.write_text(
                        json.dumps(
                            vector_metrics_raw.to_plan().to_json(),
                            indent=2,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                vector_eval = evaluate(
                    to_fpeval_plan(vector_metrics_raw.to_plan()),
                    to_fpeval_plan(target_plan),
                )
                vector_verify = verify(to_fpeval_plan(vector_metrics_raw.to_plan()))
                vector_issues = " ".join(vector_verify.issues)
                target_verify = verify(to_fpeval_plan(target_plan))
                _, _, target_furniture_valid = object_instance_targets(
                    target[index : index + 1],
                    taxonomy.furniture_ids,
                    max_instances=128,
                )
                vector_metrics = {
                    "vector_wall_f1": float(vector_eval.walls["f1"]),
                    "vector_wall_precision": float(vector_eval.walls["precision"]),
                    "vector_wall_recall": float(vector_eval.walls["recall"]),
                    "vector_wall_pred_count": float(vector_eval.walls["pred"]),
                    "vector_wall_target_count": float(vector_eval.walls["gt"]),
                    "vector_opening_f1": float(vector_eval.openings["f1"]),
                    "vector_opening_precision": float(
                        vector_eval.openings["precision"]
                    ),
                    "vector_opening_recall": float(vector_eval.openings["recall"]),
                    "vector_opening_pred_count": float(
                        vector_eval.openings["pred"]
                    ),
                    "vector_opening_target_count": float(
                        vector_eval.openings["gt"]
                    ),
                    "vector_room_f1": float(vector_eval.rooms["f1"]),
                    "vector_r_val": float(vector_eval.r_val),
                    "vector_schema_validity": float(vector_verify.valid),
                    "vector_watertight": float(vector_verify.watertight),
                    "target_watertight": float(target_verify.watertight),
                    "vector_conditional_watertight": (
                        float(vector_verify.watertight)
                        if target_verify.watertight
                        else 0.0
                    ),
                    "vector_issue_no_walls": float("no walls" in vector_issues),
                    "vector_issue_room_overlap": float(
                        "room polygons overlap" in vector_issues
                    ),
                    "vector_issue_nonpartition": float(
                        "do not tile a single connected region" in vector_issues
                    ),
                    "vector_issue_interior_gaps": float(
                        "room tiling has interior gaps" in vector_issues
                    ),
                    "vector_issue_degenerate_room": float(
                        "polygon degenerate/invalid" in vector_issues
                    ),
                    "vector_furniture_instances": float(
                        len(vector_metrics_raw.furniture)
                    ),
                    "target_furniture_instances": float(
                        target_furniture_valid.sum()
                    ),
                    **furniture_vector_metrics(
                        vector_metrics_raw.furniture,
                        vector_metrics_raw.rooms,
                        target_np,
                        taxonomy,
                    ),
                }
                predicted_wall_length = _wall_length(vector_metrics_raw.walls)
                target_wall_length = _wall_length(target_plan.walls)
                wall_length_ratio = (
                    min(predicted_wall_length, target_wall_length)
                    / max(predicted_wall_length, target_wall_length)
                    if max(predicted_wall_length, target_wall_length) > 0.0
                    else 1.0
                )
                vector_metrics.update(
                    {
                        "wall_count_abs_error_proxy": abs(
                            len(vector_metrics_raw.walls)
                            - len(target_plan.walls)
                        ),
                        "wall_length_ratio_proxy": float(wall_length_ratio),
                        "wall_endpoint_contact_rate_proxy": (
                            _wall_endpoint_contact_rate(
                                vector_metrics_raw.walls
                            )
                        ),
                        "opening_count_abs_error_proxy": abs(
                            len(vector_metrics_raw.openings)
                            - len(target_plan.openings)
                        ),
                        **_opening_match_errors(
                            vector_metrics_raw.openings,
                            target_plan.openings,
                        ),
                        "furniture_count_abs_error_proxy": abs(
                            len(vector_metrics_raw.furniture)
                            - int(target_furniture_valid.sum())
                        ),
                        **_furniture_proxy_metrics(
                            vector_metrics_raw.furniture
                        ),
                    }
                )
            instance = suite["matched_instance"]
            rows.append(
                {
                    "case": str(names[index]),
                    "room_iou": float(
                        suite["visible_semantic"]["room_macro_iou"]
                    ),
                    "instance_f1": float(
                        instance["instance_f1_at_threshold"]
                    ),
                    "instance_type_accuracy": float(
                        instance["matched_type_accuracy"]
                    ),
                    "count_abs_error": float(instance["count_abs_error"]),
                    **{key: float(value) for key, value in layer.items()},
                    **vector_metrics,
                    **{
                        f"housemind_{key}": float(value)
                        for key, value in graph.items()
                        if isinstance(
                            value,
                            (int, float, np.integer, np.floating),
                        )
                    },
                    **{
                        f"structural_{key}": float(value)
                        for key, value in structural.items()
                        if isinstance(
                            value,
                            (int, float, np.integer, np.floating),
                        )
                    },
                    **{
                        f"plausibility_{key}": float(value)
                        for key, value in plausibility.items()
                        if isinstance(value, (int, float, np.integer, np.floating))
                    },
                }
            )
    summary = {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]
        if key != "case"
    }
    watertight_eligible = [
        row for row in rows if float(row["target_watertight"]) > 0.5
    ]
    watertight_eligible_summary = {
        key: float(np.mean([row[key] for row in watertight_eligible]))
        for key in rows[0]
        if key != "case"
    }
    report = {
        "schema_version": "cad-layer-checkpoint-eval-v1",
        "checkpoint": str(cli.checkpoint),
        "test_opened": False,
        "target_used_at_inference": False,
        "summary": summary,
        "watertight_eligible_cases": len(watertight_eligible),
        "watertight_eligible_summary": watertight_eligible_summary,
        "cases": rows,
    }
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    cli.output.write_text(
        json.dumps(_jsonable(report), indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
