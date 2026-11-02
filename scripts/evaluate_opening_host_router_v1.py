#!/usr/bin/env python3
"""Evaluate routed opening intervals against the fallback opening readout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

import train_opening_host_router_v1 as train_router
from train_raster_predicted_control_bridge import RasterPredictedControlDataset

from floorflow.data.vector_primitive_target_bank import VectorPrimitiveTargetBank
from floorflow.evaluation.cad_candidate_plan import candidate_plan_metrics
from floorflow.evaluation.paired_bootstrap import paired_delta
from floorflow.heads.opening_host_router import OpeningHostRouter
from floorflow.heads.opening_query_encoder import OpeningQueryEncoder
from floorflow.training.opening_host_training import match_opening_host_targets

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer-checkpoint", type=Path, required=True)
    parser.add_argument("--scorer-checkpoint", type=Path, required=True)
    parser.add_argument("--router-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--geometry-root",
        type=Path,
        default=ROOT / "outputs/cache/cubicasa5k_geometry_v1",
    )
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_router(payload, taxonomy, device):
    config = payload["config"]
    encoder = OpeningQueryEncoder(
        int(config["field_channels"]),
        hidden_dim=int(config["hidden_dim"]),
    ).to(device)
    encoder.load_state_dict(payload["query_encoder"])
    encoder.eval()
    router = OpeningHostRouter(
        int(config["wall_hidden_dim"]),
        num_types=max(1, len(taxonomy.opening_ids)),
        route_attention=bool(config["route_attention"]),
        route_router_count=int(config["route_router_count"]),
        route_top_k=int(config["route_top_k"]),
    ).to(device)
    router.load_state_dict(payload["router"])
    router.eval()
    return encoder, router


def _case_metric_rows(
    candidates,
    scored,
    selected,
    controls,
    target,
    taxonomy,
    threshold,
    routed,
    queries,
    names,
    decoded,
):
    probability = torch.nn.functional.interpolate(
        controls["opening_family_probability"].float(),
        size=tuple(target.shape[-2:]),
        mode="bilinear",
        align_corners=False,
    )
    rows = []
    for index, name in enumerate(names):
        arguments = {
            "endpoints": candidates.endpoints[index : index + 1],
            "scores": scored["score_logits"][index : index + 1],
            "valid": candidates.valid[index : index + 1],
            "rooms": [candidates.rooms[index]],
            "host_walls": [candidates.host_walls[index]],
            "opening_probability": probability[index : index + 1],
            "target": target[index : index + 1],
            "taxonomy": taxonomy,
            "threshold": threshold,
            "wall_type_logits": scored["exterior_logits"][index : index + 1],
        }
        fallback = candidate_plan_metrics(**arguments)
        routed_row = candidate_plan_metrics(
            **arguments,
            routed_openings=[decoded[index]],
        )
        rows.append(
            {
                "case": str(name),
                **{f"fallback_{key}": value for key, value in fallback.items()},
                **{f"routed_{key}": value for key, value in routed_row.items()},
            }
        )
    return rows


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    scorer_payload = torch.load(
        args.scorer_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    train_router.scorer_payload_global = scorer_payload
    router_payload = torch.load(
        args.router_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    arguments = SimpleNamespace(
        layer_checkpoint=args.layer_checkpoint,
        scorer_checkpoint=args.scorer_checkpoint,
        geometry_root=args.geometry_root,
    )
    models = train_router.load_models(arguments, device)
    (
        _layer,
        _scorer,
        _graph_projection,
        _stage_runtime,
        config,
        _scorer_config,
        taxonomy,
    ) = models
    query_encoder, router = _load_router(router_payload, taxonomy, device)
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
        persistent_workers=args.workers > 0,
    )
    target_bank = VectorPrimitiveTargetBank(
        args.geometry_root,
        "val",
        image_size=int(config["image_size"]),
    )
    threshold = train_router._threshold(scorer_payload)
    rows = []
    host_matched_total = 0
    host_correct_total = 0
    query_total = 0
    for batch in loader:
        (
            target,
            controls,
            candidates,
            scored,
            queries,
            routed,
            selected,
            _features,
        ) = train_router._forward_batch(
            batch,
            *models,
            query_encoder,
            router,
            device,
        )
        decoded = train_router._routed_rows(
            routed,
            selected,
            candidates,
            taxonomy,
            float(target.shape[-1]),
            queries,
        )
        case_rows = _case_metric_rows(
            candidates,
            scored,
            selected,
            controls,
            target,
            taxonomy,
            threshold,
            routed,
            queries,
            batch[3],
            decoded,
        )
        targets = target_bank.batch_targets(batch[3], device)
        host_targets = match_opening_host_targets(
            queries,
            candidates,
            targets,
            taxonomy,
            candidate_mask=selected,
        )
        host_matched_total += int(host_targets.matched.sum())
        query_total += int(queries.valid.sum())
        host_correct_total += int(
            (
                routed["host_index"][host_targets.matched.to(device)]
                == host_targets.hosts.to(device)[host_targets.matched.to(device)]
            ).sum()
        )
        for index, row in enumerate(case_rows):
            matched = host_targets.matched[index]
            row["host_accuracy"] = (
                float(
                    (
                        routed["host_index"][index][matched]
                        == host_targets.hosts[index].to(device)[matched]
                    )
                    .float()
                    .mean()
                )
                if bool(matched.any())
                else 1.0
            )
        rows.extend(case_rows)
    metrics = sorted(
        key
        for key in rows[0]
        if key != "case"
    )
    summary = {
        key: float(np.mean([row[key] for row in rows]))
        for key in metrics
    }
    summary["host_accuracy_weighted"] = host_correct_total / max(
        1,
        host_matched_total,
    )
    summary["host_matched_queries"] = host_matched_total
    summary["host_query_match_rate"] = host_matched_total / max(1, query_total)
    paired = {}
    for key in ("wall", "opening", "room", "eligible_opening"):
        routed_rows = [
            {"value": row[f"routed_{key}"]}
            for row in rows
        ]
        fallback_rows = [
            {"value": row[f"fallback_{key}"]}
            for row in rows
        ]
        paired[key] = paired_delta(
            routed_rows,
            fallback_rows,
            "value",
            iterations=5000,
            seed=17,
        )
    gate = {
        "opening_f1_at_least_0_35": summary["routed_opening"] >= 0.35,
        "host_accuracy_weighted_at_least_0_80": (
            summary["host_accuracy_weighted"] >= 0.8
        ),
        "host_query_match_rate_at_least_0_25": (
            summary["host_query_match_rate"] >= 0.25
        ),
        "validity_one": summary["routed_valid"] == 1.0,
        "watertight_one": summary["routed_watertight"] == 1.0,
        "opening_paired_ci_lower_positive": paired["opening"]["ci_low"] > 0.0,
        "wall_unchanged": paired["wall"]["mean_delta"] == 0.0,
        "room_unchanged": paired["room"]["mean_delta"] == 0.0,
    }
    gate = {key: bool(value) for key, value in gate.items()}
    gate["all_passed"] = all(gate.values())
    output = {
        "schema_version": "opening-host-router-eval-v1",
        "scope": "final selected-wall hosts and learned opening intervals",
        "test_opened": False,
        "target_used_at_inference": False,
        "cases": len(rows),
        "summary": summary,
        "paired_routed_vs_fallback": paired,
        "publication_gate": gate,
        "per_case": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "publication_gate": gate}, indent=2))


if __name__ == "__main__":
    main()
