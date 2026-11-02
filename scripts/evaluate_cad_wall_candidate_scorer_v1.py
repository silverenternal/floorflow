#!/usr/bin/env python3
"""Audit score selection versus endpoint residual for wall candidates."""

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
from train_vector_topology_scaffold_v1 import GRAPH_STAGES

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.cad_candidate_plan import candidate_plan_metrics
from floorflow.evaluation.cad_plan_metrics import cad_wall_plan_metrics
from floorflow.evaluation.paired_bootstrap import paired_delta
from floorflow.heads.cad_wall_candidate_scorer import CadWallCandidateScorer
from floorflow.inference.cad_stage_runtime import CadStageRuntime
from floorflow.inference.cad_wall_candidate_runtime import (
    compose_scored_candidate_plan,
    select_opening_safe_plan,
)
from floorflow.model.predicted_channel_checkpoint import (
    build_predicted_channel_model,
    load_predicted_channel_state_dict,
)
from floorflow.training.cad_wall_candidate_epoch_cache import load_candidate_batch
from floorflow.training.cad_wall_candidate_training import build_candidate_batch

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer-checkpoint", type=Path, required=True)
    parser.add_argument("--scorer-checkpoint", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _wall_opening_evidence(
    endpoints: torch.Tensor,
    opening_probability: torch.Tensor,
    *,
    samples: int = 17,
) -> torch.Tensor:
    probability = opening_probability.float().amax(dim=1, keepdim=True)
    fraction = torch.linspace(
        0.0,
        1.0,
        samples,
        device=endpoints.device,
    )
    points = (
        endpoints[:, :, None, 0] * (1.0 - fraction[None, None, :, None])
        + endpoints[:, :, None, 1] * fraction[None, None, :, None]
    )
    grid = points * 2.0 - 1.0
    sampled = F.grid_sample(
        probability,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled[:, 0].mean(dim=-1)


def _plan_metric_row(plan, target: np.ndarray, taxonomy) -> dict[str, float]:
    return cad_wall_plan_metrics(plan, target, taxonomy)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    layer_payload = torch.load(
        args.layer_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    scorer_payload = torch.load(
        args.scorer_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    config = layer_payload["config"]
    scorer_config = scorer_payload["config"]
    taxonomy = FloorplanTaxonomy.from_json(layer_payload["taxonomy"])
    layer_model = build_predicted_channel_model(taxonomy, config).to(device)
    load_predicted_channel_state_dict(layer_model, layer_payload["model"])
    layer_model.eval()
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
    scorer = CadWallCandidateScorer(
        int(config["field_channels"]),
        hidden_dim=int(scorer_config["hidden_dim"]),
        graph_dim=int(scorer_config["hidden_dim"]),
        residual_scale=float(scorer_config.get("residual_scale", 0.01)),
        use_encoded_score=bool(scorer_config.get("use_encoded_score", False)),
        source_features=(
            str(scorer_config.get("candidate_mode", "boundary"))
            == "skeleton_portfolio"
        ),
    ).to(device)
    scorer.load_state_dict(scorer_payload["scorer"])
    scorer.eval()
    graph_projection = torch.nn.Linear(
        int(config["field_channels"]),
        int(scorer_config["hidden_dim"]),
    ).to(device)
    graph_projection.load_state_dict(scorer_payload["graph_projection"])
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
        pin_memory=True,
    )
    selection = str(scorer_payload.get("selection_mode", "threshold=-1.0"))
    threshold = (
        float(selection.split("=", 1)[1])
        if selection.startswith("threshold=")
        else -1.0
    )
    candidate_cache_root = scorer_config.get("candidate_cache_root")
    candidate_cache = {}
    case_rows: dict[str, list[dict]] = {}
    count = 0
    class Args:
        predicted_root = Path(config["predicted_root"])
        room_size = 224
        structure_size = 128

    for target, footprint, floor_ids, names, image in loader:
        target = target.to(device)
        footprint = footprint.to(device)
        if candidate_cache_root is not None:
            candidates, tokens, opening_probability, field_features = (
                load_candidate_batch(
                    cache=candidate_cache,
                    split="val",
                    names=names,
                    root=Path(config["predicted_root"]),
                    args=Args(),
                    device=device,
                    footprint=footprint,
                    image=image,
                    floor_ids=floor_ids,
                    taxonomy=taxonomy,
                    stage_runtime=stage_runtime,
                    layer_model=layer_model,
                    controls_loader=_predicted_controls,
                    graph_stages=GRAPH_STAGES,
                    minimum_room_area=int(scorer_config["minimum_room_area"]),
                    max_walls=int(scorer_config["max_walls"]),
                    disk_root=Path(candidate_cache_root),
                )
            )
        else:
            controls = _predicted_controls(
                Path(config["predicted_root"]),
                "val",
                names,
                Args(),
                device,
            )
            controls, _ = stage_runtime(
                controls,
                footprint,
                image.to(device),
                floor_ids.to(device),
            )
            output = layer_model(controls, footprint)
            tokens = torch.stack(
                [output["action_tokens"][name] for name in GRAPH_STAGES],
                dim=1,
            )
            field_features = output["field_features"]
            candidates = build_candidate_batch(
                controls,
                output,
                footprint,
                taxonomy,
                minimum_room_area=int(scorer_config["minimum_room_area"]),
                max_walls=int(scorer_config["max_walls"]),
                device=device,
                candidate_mode=scorer_config.get("candidate_mode", "boundary"),
            )
            opening_probability = torch.nn.functional.interpolate(
                controls["opening_family_probability"].float(),
                size=tuple(target.shape[-2:]),
                mode="bilinear",
                align_corners=False,
            )
        scored = scorer(
            field_features,
            candidates.endpoints,
            candidates.valid,
            candidates.evidence,
            graph_projection(tokens),
            candidates.exterior,
            proposal_source=candidates.proposal_source,
        )
        batch_size = int(target.shape[0])
        arms = {
            "control": (
                candidates.control_endpoints,
                candidates.control_evidence,
                0.2,
                None,
                None,
                None,
            ),
            "raw_score": (
                candidates.endpoints,
                scored["score_logits"],
                threshold,
                None,
                scored["exterior_logits"],
                None,
            ),
            "refined_score": (
                scored["endpoints"],
                scored["score_logits"],
                threshold,
                None,
                scored["exterior_logits"],
                None,
            ),
            "raw_budget": (
                candidates.endpoints,
                scored["score_logits"],
                0.0,
                scored["budget_count"].cpu().round().long(),
                scored["exterior_logits"],
                None,
            ),
            "refined_budget": (
                scored["endpoints"],
                scored["score_logits"],
                0.0,
                scored["budget_count"].cpu().round().long(),
                scored["exterior_logits"],
                None,
            ),
        }
        source_arms = []
        if candidates.proposal_source is not None:
            for source in (2, 3, 4, 5):
                arm = f"source_{source}"
                arms[arm] = (
                    candidates.endpoints,
                    torch.where(
                        candidates.proposal_source == source,
                        torch.ones_like(scored["score_logits"]),
                        torch.full_like(scored["score_logits"], -1.0),
                    ),
                    0.5,
                    None,
                    None,
                    None,
                )
                source_arms.append(arm)
        control_plans = [
            compose_scored_candidate_plan(
                candidates.control_endpoints[index],
                candidates.control_evidence[index],
                candidates.control_valid[index],
                candidates.rooms[index],
                float(target.shape[-1]),
                candidates.host_walls[index],
                opening_probability[index].cpu().numpy(),
                taxonomy,
                score_threshold=0.2,
            )
            for index in range(batch_size)
        ]
        control_openings = [plan.openings for plan in control_plans]
        arms["raw_score_preserve_openings"] = (
            candidates.endpoints,
            scored["score_logits"],
            threshold,
            None,
            scored["exterior_logits"],
            control_openings,
        )
        opening_evidence = _wall_opening_evidence(
            candidates.endpoints,
            opening_probability,
        )
        for weight in (1.0, 2.0, 4.0):
            arms[f"raw_score_opening_{weight:g}"] = (
                candidates.endpoints,
                scored["score_logits"] + float(weight) * opening_evidence,
                threshold,
                None,
                scored["exterior_logits"],
                None,
            )
        for name, (
            endpoints,
            scores,
            cut,
            budget,
            type_logits,
            routed_openings,
        ) in arms.items():
            rows = case_rows.setdefault(name, [])
            for index in range(batch_size):
                metrics = candidate_plan_metrics(
                    endpoints[index : index + 1],
                    scores[index : index + 1],
                    candidates.valid[index : index + 1],
                    [candidates.rooms[index]],
                    [candidates.host_walls[index]],
                    opening_probability[index : index + 1],
                    target[index : index + 1],
                    taxonomy,
                    cut,
                    wall_budget=(
                        budget[index : index + 1]
                        if torch.is_tensor(budget)
                        else budget
                    ),
                    wall_type_logits=(
                        None
                        if type_logits is None
                        else type_logits[index : index + 1]
                    ),
                    routed_openings=(
                        None
                        if routed_openings is None
                        else [routed_openings[index]]
                    ),
                )
                rows.append(
                    {"case": str(names[index]), **metrics}
                )
        safe_rows = case_rows.setdefault("raw_score_opening_safe_gate", [])
        for index in range(batch_size):
            candidate_plan = compose_scored_candidate_plan(
                candidates.endpoints[index],
                scored["score_logits"][index],
                candidates.valid[index],
                candidates.rooms[index],
                float(target.shape[-1]),
                candidates.host_walls[index],
                opening_probability[index].cpu().numpy(),
                taxonomy,
                score_threshold=threshold,
                wall_type_logits=scored["exterior_logits"][index],
            )
            safe_plan = select_opening_safe_plan(
                control_plans[index],
                candidate_plan,
            )
            plan_metrics = _plan_metric_row(
                safe_plan,
                target[index].cpu().numpy(),
                taxonomy,
            )
            safe_rows.append(
                {
                    "case": str(names[index]),
                    **plan_metrics,
                }
            )
            control_row = next(
                row
                for row in case_rows["control"]
                if row["case"] == str(names[index])
            )
            control_row.update(
                _plan_metric_row(
                    control_plans[index],
                    target[index].cpu().numpy(),
                    taxonomy,
                )
            )
        count += batch_size
    summary_keys = sorted(
        {
            key
            for rows in case_rows.values()
            for key in rows[0]
            if key != "case"
        }
    )
    summary = {
        arm: {
            key: float(
                np.mean(
                    [
                        row[key]
                        for row in case_rows[arm]
                        if isinstance(row.get(key), (int, float))
                    ]
                )
            )
            for key in summary_keys
            if any(
                isinstance(row.get(key), (int, float))
                for row in case_rows[arm]
            )
        }
        for arm in case_rows
    }
    paired = {}
    for arm in (
        "raw_score",
        "refined_score",
        "raw_budget",
        "refined_budget",
        "raw_score_preserve_openings",
        "raw_score_opening_1",
        "raw_score_opening_2",
        "raw_score_opening_4",
        "raw_score_opening_safe_gate",
        *source_arms,
    ):
        common_keys = (
            set(case_rows[arm][0]) & set(case_rows["control"][0])
        ) - {"case"}
        paired[arm] = {
            key: paired_delta(
                case_rows[arm],
                case_rows["control"],
                key,
                iterations=5000,
                seed=17,
            )
            for key in (
                "wall",
                "wall_precision",
                "wall_recall",
                "junction_f1",
                "wall_continuity",
                "opening",
                "room",
                "eligible_wall",
            )
            if key in common_keys
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected = summary["raw_score_opening_safe_gate"]
    selected_paired = paired["raw_score_opening_safe_gate"]
    publication_gate = {
        "wall_f1_at_least_0_58": selected["wall"] >= 0.58,
        "wall_precision_at_least_0_52": selected["wall_precision"] >= 0.52,
        "wall_recall_at_least_0_65": selected["wall_recall"] >= 0.65,
        "junction_f1_at_least_0_65": selected["junction_f1"] >= 0.65,
        "wall_continuity_at_least_0_99": selected["wall_continuity"] >= 0.99,
        "validity_one": selected["valid"] == 1.0,
        "watertight_one": selected["watertight"] == 1.0,
        "opening_unchanged": selected_paired["opening"]["mean_delta"] == 0.0,
        "room_unchanged": selected_paired["room"]["mean_delta"] == 0.0,
    }
    publication_gate["all_passed"] = all(publication_gate.values())
    args.output.write_text(
        json.dumps(
            {
                "schema_version": "cad-wall-candidate-scorer-audit-v1",
                "scorer_checkpoint": str(args.scorer_checkpoint),
                "selection_mode": selection,
                "cases": count,
                "test_opened": False,
                "target_used_at_inference": False,
                "summary": summary,
                "paired_vs_control": paired,
                "publication_gate": publication_gate,
                "per_case": case_rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
