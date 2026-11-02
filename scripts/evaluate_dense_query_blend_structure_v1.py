#!/usr/bin/env python3
"""Evaluate dense/query blend through the complete predicted CAD consumer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_raster_control_router_screen import _decode_layers, _structure_outputs
from train_raster_predicted_control_bridge import RasterPredictedControlDataset
from train_residential_predicted_refiner import _predicted_controls

from floorflow.data.control_contract import sha256_file
from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.paired_bootstrap import paired_delta
from floorflow.evaluation.predicted_control_fidelity import (
    predicted_control_fidelity_metrics,
)
from floorflow.evaluation.predicted_control_reporting import numeric_row_summary
from floorflow.evaluation.wall_false_positive import (
    wall_false_positive_decomposition,
)
from floorflow.heads.dense_query_support_gate import (
    SelectiveRoomSupportDeletion,
    render_support_conditioned_room_labels,
)
from floorflow.heads.joint_control_loading import load_joint_control_repair
from floorflow.heads.raster_control_bank import (
    load_raster_instance_control_model,
    load_raster_room_model,
)
from floorflow.inference.dense_query_eval_cli import parse_args
from floorflow.inference.opening_heatmap_decoder import (
    load_opening_heatmap_decoder,
    predict_opening_family_probabilities,
)
from floorflow.inference.structure_compositor import (
    StructureReadoutPolicy,
    compose_structure_readout,
    owner_cell_plan,
    structure_opening_family_outputs,
)
from floorflow.inference.support_structure_fusion import (
    apply_room_support_to_visible_plan,
)
from floorflow.representation.plan_canonical_adapter import (
    canonical_token_roundtrip,
    raster_plan_to_canonical,
)
from floorflow.training.raster_instance_evaluation import resolve_query_blend

ROOT = Path(__file__).resolve().parents[1]
PAIRED_KEYS = (
    "room_macro_iou",
    "room_support_iou",
    "instance_instance_f1_at_threshold",
    "instance_count_abs_error",
    "instance_matched_type_accuracy",
    "fpeval_wall_f1",
    "fpeval_opening_f1",
    "node_f1",
    "edge_overlap",
    "ged",
    "validity",
    "functional_watertight",
    "wall_continuity_rate",
    "outside_footprint_fraction",
)
LOWER_IS_BETTER = {"instance_count_abs_error", "ged", "outside_footprint_fraction"}


def _load_structure_checkpoint(path: Path, taxonomy, device: torch.device):
    model, config = load_joint_control_repair(path, taxonomy, device)
    return (
        model,
        {
            "room_size": int(config.get("room_size", 224)),
            "structure_size": int(config.get("structure_size", 128)),
            "structure_contract": str(
                config.get("structure_contract", "legacy")
            ),
        },
    )


def _gate_report(summary: dict, paired: dict, candidate_arm: str) -> dict[str, object]:
    candidate = summary[candidate_arm]
    validity = float(candidate["validity_mean"])
    watertight = float(candidate["functional_watertight_mean"])
    outside = float(candidate["outside_footprint_fraction_mean"])
    primary = paired["room_macro_iou"]["mean_delta"] > 0.0
    support = paired["room_support_iou"]["mean_delta"] >= 0.0
    instance_f1 = (
        paired["instance_instance_f1_at_threshold"]["mean_delta"] >= 0.0
    )
    count = paired["instance_count_abs_error"]["mean_delta"] <= 0.0
    wall = paired["fpeval_wall_f1"]["mean_delta"] >= 0.0
    node = paired["node_f1"]["mean_delta"] >= 0.0
    edge = paired["edge_overlap"]["mean_delta"] >= 0.0
    ged = paired["ged"]["mean_delta"] <= 0.0
    return {
        "room_iou_non_regression": bool(primary),
        "room_support_non_regression": bool(support),
        "instance_f1_non_regression": bool(instance_f1),
        "count_mae_non_regression": bool(count),
        "wall_f1_non_regression": bool(wall),
        "node_f1_non_regression": bool(node),
        "edge_overlap_non_regression": bool(edge),
        "ged_non_regression": bool(ged),
        "validity_one": bool(validity == 1.0),
        "functional_watertight_at_least_0_99": bool(watertight >= 0.99),
        "outside_footprint_zero": bool(outside <= 1.0e-12),
        "passed": bool(
            primary
            and support
            and instance_f1
            and count
            and wall
            and node
            and edge
            and ged
            and validity == 1.0
            and watertight >= 0.99
            and outside <= 1.0e-12
        ),
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    taxonomy = FloorplanTaxonomy.read(args.cache_root / "taxonomy.json")
    dataset = RasterPredictedControlDataset(
        args.predicted_root,
        args.split,
        args.cache_root,
        args.data_root,
        image_size=args.image_size,
        limit=args.limit,
        offset=args.offset,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )
    dense_model = load_raster_room_model(
        args.dense_checkpoint,
        taxonomy,
        device,
    )
    query_model = load_raster_instance_control_model(
        args.query_checkpoint,
        taxonomy,
        device,
    )
    opening_runtime = (
        load_opening_heatmap_decoder(
            args.opening_heatmap_checkpoint,
            taxonomy,
            device,
        )
        if args.opening_heatmap_checkpoint is not None
        else None
    )
    query_weight, query_weight_source = resolve_query_blend(
        args.query_checkpoint,
        args.query_weight,
    )
    args.query_weight = query_weight
    readout_policy = StructureReadoutPolicy(
        wall_threshold=args.readout_wall_threshold,
        wall_core_threshold=args.readout_wall_core_threshold,
        wall_tail_radius=args.readout_wall_tail_radius,
        opening_threshold=args.readout_opening_threshold,
        junction_threshold=args.readout_junction_threshold,
        junction_degree_min=args.readout_junction_degree_min,
        seam_mode=args.readout_seam_mode,
        room_boundary_threshold=args.readout_room_boundary_threshold,
        opening_support_radius=args.readout_opening_support_radius,
        opening_min_area=args.readout_opening_min_area,
        exterior_mode=args.readout_exterior_mode,
        exterior_width=args.boundary_width,
        reference_resolution=args.image_size,
    )
    readout_policy.validate()
    support_gate = None
    if args.support_gate_checkpoint is not None:
        gate_payload = torch.load(
            args.support_gate_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        gate_config = gate_payload["config"]
        support_gate = SelectiveRoomSupportDeletion(
            len(taxonomy.room_ids),
            fixed_blend=float(gate_config["fixed_blend"]),
            hidden_channels=int(gate_config["hidden_channels"]),
            preserve_confidence=float(gate_config["preserve_confidence"]),
        ).to(device)
        support_gate.load_state_dict(gate_payload["model"])
        support_gate.eval()
    structure_model, structure_sizes = _load_structure_checkpoint(
        args.structure_checkpoint,
        taxonomy,
        device,
    )
    runtime_args = argparse.Namespace(
        **vars(args),
        **structure_sizes,
    )
    rows: dict[str, list[dict]] = (
        {"blend": []}
        if args.skip_dense
        else {"dense": [], "blend": []}
    )
    if support_gate is not None:
        rows["selective"] = []
    processed = 0
    for target, footprint, floor_ids, names, image in loader:
        target = target.to(device)
        footprint = footprint.to(device)
        floor_ids = floor_ids.to(device)
        image = image.to(device)
        predicted = _predicted_controls(
            args.predicted_root,
            args.split,
            names,
            runtime_args,
            device,
        )
        dense_logits = dense_model(
            footprint,
            control_image=image,
        )
        query_output = query_model(footprint, control_image=image)
        query_logits = query_output["fused_logits"]
        blend_probability = (
            (1.0 - query_weight) * dense_logits.softmax(dim=1)
            + query_weight * query_logits.softmax(dim=1)
        )
        blend_logits = blend_probability.clamp_min(1.0e-8).log()
        opening_probability = (
            predict_opening_family_probabilities(
                opening_runtime,
                footprint,
                image,
                taxonomy,
            )
            if opening_runtime is not None
            else None
        )
        labels_by_arm = {}
        structure_openings_by_arm = {}
        structure_arms = [("blend", blend_logits)]
        if "dense" in rows:
            structure_arms.insert(0, ("dense", dense_logits))
        for arm, room_logits in structure_arms:
            room_logits = F.interpolate(
                room_logits,
                size=(runtime_args.room_size, runtime_args.room_size),
                mode="bilinear",
                align_corners=False,
            )
            structure = _structure_outputs(
                structure_model,
                predicted,
                room_logits,
                footprint,
                runtime_args,
                repair_room=args.repair_room,
            )["structure"]
            layers, _rooms, _wall, _opening = _decode_layers(
                room_logits,
                structure,
                footprint,
                taxonomy,
                policy=readout_policy,
            )
            labels_by_arm[arm] = compose_structure_readout(
                layers,
                taxonomy,
                policy=readout_policy,
            )
            if args.structure_opening_interval_readout:
                structure_openings_by_arm[arm] = (
                    structure_opening_family_outputs(
                        structure,
                        footprint.shape[-2:],
                    )
                )
        if support_gate is not None:
            gate_output = support_gate(
                dense_logits.softmax(dim=1),
                query_logits.softmax(dim=1),
                query_output["mask_logits"].sigmoid().max(dim=1).values,
                footprint,
                keep_threshold=float(
                    gate_config.get("keep_threshold", 0.5)
                ),
            )
            room_ids = torch.tensor(taxonomy.room_ids, device=device)
            gated_rooms = render_support_conditioned_room_labels(
                blend_probability,
                gate_output["keep_mask"],
                room_ids,
                footprint,
                topology_safe=args.topology_safe,
                minimum_deletable_component_area=(
                    args.min_deletable_component_area
                ),
            )
            labels_by_arm["selective"] = apply_room_support_to_visible_plan(
                labels_by_arm["blend"],
                gated_rooms > 0,
                room_ids,
            )
        for index, name in enumerate(names):
            for arm, labels in labels_by_arm.items():
                prediction_plan = (
                    owner_cell_plan(
                        labels[index].cpu().numpy(),
                        footprint[index].cpu().numpy(),
                        taxonomy,
                    )
                    if args.owner_cell_plan
                    else None
                )
                token_metrics = {}
                if args.canonical_token_readout:
                    canonical_plan = prediction_plan or owner_cell_plan(
                        labels[index].cpu().numpy(),
                        footprint[index].cpu().numpy(),
                        taxonomy,
                    )
                    try:
                        canonical, _ = raster_plan_to_canonical(
                            canonical_plan,
                            footprint[index].cpu().numpy(),
                            taxonomy,
                            case_id=str(name),
                            track="cubicasa_reconstruction",
                        )
                        _tokenized, token_metrics = (
                            canonical_token_roundtrip(
                                canonical,
                                taxonomy,
                            )
                        )
                        token_metrics["canonical_readout_success"] = 1.0
                    except (RuntimeError, ValueError):
                        token_metrics = {
                            "canonical_readout_success": 0.0,
                            "canonical_fingerprint_exact": False,
                        }
                rows[arm].append(
                    {
                        "case": str(name),
                        "index": int(args.offset) + len(rows[arm]),
                        **wall_false_positive_decomposition(
                            labels[index].cpu().numpy(),
                            target[index].cpu().numpy(),
                            footprint[index].cpu().numpy(),
                            taxonomy,
                        ),
                        **predicted_control_fidelity_metrics(
                            labels[index].cpu().numpy(),
                            target[index].cpu().numpy(),
                            footprint[index].cpu().numpy(),
                            taxonomy,
                            reconcile_gap_fraction=args.reconcile_gap_fraction,
                            fpeval_tolerance=args.fpeval_tolerance,
                            graph_ged_timeout_seconds=(
                                args.ged_timeout_seconds
                            ),
                            graph_source=args.graph_source,
                            prediction_plan=prediction_plan,
                            **token_metrics,
                            **(
                                {
                                    "prediction_opening_heatmap": (
                                        opening_probability[index]
                                        .cpu()
                                        .numpy()
                                    ),
                                    "prediction_opening_heatmap_threshold": (
                                        args.opening_heatmap_threshold
                                    ),
                                    "prediction_opening_min_area": (
                                        args.opening_heatmap_min_area
                                    ),
                                    "prediction_opening_max_wall_distance": (
                                        args.opening_heatmap_max_wall_distance
                                    ),
                                }
                                if opening_probability is not None
                                else {}
                            ),
                            **(
                                {
                                    "prediction_opening_probability": (
                                        structure_openings_by_arm[arm][0][
                                            index
                                        ].cpu().numpy()
                                    ),
                                    "prediction_opening_logits": (
                                        structure_openings_by_arm[arm][1][
                                            index
                                        ].cpu().numpy()
                                    ),
                                    "prediction_opening_threshold": float(
                                        args.readout_opening_threshold
                                    ),
                                }
                                if (
                                    opening_probability is None
                                    and args.structure_opening_interval_readout
                                )
                                else {}
                            ),
                        ),
                    }
                )
        processed += len(names)
        if processed % 40 == 0 or processed == len(dataset):
            print(
                json.dumps(
                    {
                        "progress_cases": processed,
                        "total_cases": len(dataset),
                    }
                ),
                flush=True,
            )
    summary = {arm: numeric_row_summary(values) for arm, values in rows.items()}
    candidate_arm = "selective" if support_gate is not None else "blend"
    baseline_arm = "blend" if support_gate is not None else "dense"
    paired = (
        {
            key: paired_delta(
                rows[candidate_arm],
                rows[baseline_arm],
                key,
                iterations=args.iterations,
                seed=args.seed,
                lower_is_better=key in LOWER_IS_BETTER,
            )
            for key in PAIRED_KEYS
        }
        if baseline_arm in rows
        else {}
    )
    gates = (
        _gate_report(summary, paired, candidate_arm)
        if paired
        else {
            "validity_one": bool(
                summary[candidate_arm]["validity_mean"] == 1.0
            ),
            "functional_watertight_at_least_0_99": bool(
                summary[candidate_arm]["functional_watertight_mean"] >= 0.99
            ),
            "outside_footprint_zero": bool(
                summary[candidate_arm]["outside_footprint_fraction_mean"]
                <= 1.0e-12
            ),
        }
    )
    if not paired:
        gates["passed"] = all(gates.values())
    report = {
        "schema_version": "dense-query-blend-structure-v1",
        "protocol": "predicted_control",
        "split": args.split,
        "validation_cases": len(dataset),
        "validation_offset": args.offset,
        "test_opened": False,
        "target_used_at_inference": False,
        "config": {
            "predicted_root": str(args.predicted_root),
            "structure_checkpoint": str(args.structure_checkpoint),
            "structure_checkpoint_sha256": sha256_file(
                args.structure_checkpoint
            ),
            "dense_checkpoint": str(args.dense_checkpoint),
            "dense_checkpoint_sha256": sha256_file(args.dense_checkpoint),
            "query_checkpoint": str(args.query_checkpoint),
            "query_checkpoint_sha256": sha256_file(args.query_checkpoint),
            "support_gate_checkpoint": (
                str(args.support_gate_checkpoint)
                if args.support_gate_checkpoint is not None
                else None
            ),
            "support_gate_checkpoint_sha256": (
                sha256_file(args.support_gate_checkpoint)
                if args.support_gate_checkpoint is not None
                else None
            ),
            "opening_heatmap_checkpoint": (
                str(args.opening_heatmap_checkpoint)
                if args.opening_heatmap_checkpoint is not None
                else None
            ),
            "opening_heatmap_checkpoint_sha256": (
                sha256_file(args.opening_heatmap_checkpoint)
                if args.opening_heatmap_checkpoint is not None
                else None
            ),
            "opening_heatmap_threshold": args.opening_heatmap_threshold,
            "opening_heatmap_min_area": args.opening_heatmap_min_area,
            "opening_heatmap_max_wall_distance": (
                args.opening_heatmap_max_wall_distance
            ),
            "topology_safe": args.topology_safe,
            "min_deletable_component_area": (
                args.min_deletable_component_area
            ),
            "skip_dense": args.skip_dense,
            "taxonomy_sha256": sha256_file(args.cache_root / "taxonomy.json"),
            "query_weight": args.query_weight,
            "query_weight_source": query_weight_source,
            "repair_room": args.repair_room,
            "readout_policy": dict(readout_policy.__dict__),
            "owner_cell_plan": args.owner_cell_plan,
            "structure_opening_interval_readout": (
                args.structure_opening_interval_readout
            ),
            "graph_source": args.graph_source,
            "canonical_token_readout": args.canonical_token_readout,
            "image_size": args.image_size,
            **structure_sizes,
            "boundary_width": args.boundary_width,
            "reconcile_gap_fraction": args.reconcile_gap_fraction,
        },
        "summary": summary,
        "comparison": {
            "candidate_arm": candidate_arm,
            "baseline_arm": baseline_arm if paired else None,
        },
        "paired": paired,
        "gates": gates,
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "summary": {
                    arm: {
                        key: value
                        for key, value in values.items()
                        if key.endswith("_mean")
                        and key
                        in {
                            "room_macro_iou_mean",
                            "instance_instance_f1_at_threshold_mean",
                            "fpeval_wall_f1_mean",
                            "fpeval_opening_f1_mean",
                            "functional_watertight_mean",
                            "outside_footprint_fraction_mean",
                        }
                    }
                    for arm, values in summary.items()
                },
                "paired": paired,
                "gates": gates,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
