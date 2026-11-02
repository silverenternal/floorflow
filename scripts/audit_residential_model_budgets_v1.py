#!/usr/bin/env python3
"""Audit residential parameter and training budgets without running a model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.heads.joint_control_repair import JointControlRepair
from floorflow.heads.opening_host_router import OpeningHostRouter
from floorflow.heads.raster_control_bank import RasterInstanceControl
from floorflow.model.predicted_channel_checkpoint import (
    build_predicted_channel_model,
)

ROOT = Path(__file__).resolve().parents[1]


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _budget(train_cases: int, epochs: int, batch_size: int) -> dict[str, int]:
    updates = epochs * ((int(train_cases) + int(batch_size) - 1) // int(batch_size))
    return {
        "train_cases": int(train_cases),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "sample_exposures": int(train_cases) * int(epochs),
        "optimizer_updates": int(updates),
    }


def _predicted_model(
    taxonomy: FloorplanTaxonomy,
    config: dict,
) -> torch.nn.Module:
    model = dict(config)
    model["grid"] = int(model.get("grid", model.get("field_grid", 128)))
    return build_predicted_channel_model(taxonomy, model)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--room-checkpoint",
        type=Path,
        default=(
            ROOT
            / "outputs/train/room_pagefloor_confirmation_full5122_v1/seed29/best.pt"
        ),
    )
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=ROOT / "outputs/cache/cubicasa5k_full_v2_384/taxonomy.json",
    )
    parser.add_argument(
        "--static-channel-config",
        type=Path,
        default=ROOT / "configs/typed_channel_attention_v1.json",
    )
    parser.add_argument(
        "--dynamic-channel-config",
        type=Path,
        default=ROOT / "configs/predicted_channel_source_router_v3.json",
    )
    parser.add_argument(
        "--balanced-budget-config",
        type=Path,
        default=ROOT / "configs/weak_metric_balanced_budget_v1.json",
    )
    parser.add_argument(
        "--room-candidate-config",
        type=Path,
        default=ROOT / "configs/room_query_head_mita_v3.json",
    )
    parser.add_argument(
        "--opening-router-config",
        type=Path,
        default=ROOT / "configs/opening_host_mita_v2.json",
    )
    parser.add_argument(
        "--structure-checkpoint",
        type=Path,
        default=(
            ROOT
            / "outputs/train/raster_structure_bridge_full_seed23_v1/best.pt"
        ),
    )
    parser.add_argument(
        "--structure-candidate-config",
        type=Path,
        default=ROOT / "configs/structure_bridge_mita_v3.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/residential_model_budget_audit_v1.json",
    )
    args = parser.parse_args()

    taxonomy = FloorplanTaxonomy.read(args.taxonomy)
    room_payload = torch.load(
        args.room_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    room_state = room_payload["model"]
    room_groups = {
        name: sum(
            value.numel()
            for key, value in room_state.items()
            if key.startswith(name)
        )
        for name in ("room.", "mask_decoder.", "query_set_head.")
    }
    room_total = sum(value.numel() for value in room_state.values())
    room_candidate_config = _read(args.room_candidate_config)["model"]
    room_candidate_config.pop("estimated_parameters", None)
    room_candidate_config["num_queries"] = room_candidate_config.pop(
        "queries"
    )
    room_candidate = RasterInstanceControl(
        taxonomy,
        **room_candidate_config,
    )
    room_candidate_groups = {
        name: sum(
            parameter.numel()
            for key, parameter in room_candidate.named_parameters()
            if key.startswith(f"{name}.")
        )
        for name in ("room", "mask_decoder", "query_set_head")
    }
    room_candidate_total = _count_parameters(room_candidate)

    static_config = _read(args.static_channel_config)
    static_model = _predicted_model(
        taxonomy,
        {
            "control_channels": [24, 38, 52, 28],
            "program_channels": 24,
            "grid": static_config["field_grid"],
            "field_channels": static_config["field_channels"],
            "head_channels": static_config["head_channels"],
            "attention_grid": static_config["attention_grid"],
            "layers": static_config["layers"],
            "full_resolution_output": True,
            "typed_channel_links": True,
            "control_evidence_context": False,
            "dynamic_channel_links": False,
        },
    )
    dynamic_config = _read(args.dynamic_channel_config)
    dynamic_model = _predicted_model(taxonomy, dynamic_config["model"])
    balanced_config = _read(args.balanced_budget_config)
    structure_payload = torch.load(
        args.structure_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    structure_params = sum(
        value.numel() for value in structure_payload["model"].values()
    )
    structure_candidate_config = _read(
        args.structure_candidate_config
    )["model"]
    structure_candidate = JointControlRepair(
        len(taxonomy.room_ids),
        len(taxonomy.opening_ids),
        room_base_channels=int(
            structure_candidate_config["room_base_channels"]
        ),
        structure_base_channels=int(
            structure_candidate_config["structure_base_channels"]
        ),
        structure_contract=str(
            structure_candidate_config["structure_contract"]
        ),
        host_conditioning_v2=bool(
            structure_candidate_config.get(
                "host_conditioning_v2",
                False,
            )
        ),
        structure_global_route_attention=bool(
            structure_candidate_config["structure_global_route_attention"]
        ),
        structure_route_grid=int(
            structure_candidate_config["structure_route_grid"]
        ),
        structure_route_router_count=int(
            structure_candidate_config["structure_route_router_count"]
        ),
        structure_route_top_k=int(
            structure_candidate_config["structure_route_top_k"]
        ),
        structure_route_heads=int(
            structure_candidate_config["structure_route_heads"]
        ),
    )
    structure_candidate_groups = {
        "room_refiner": _count_parameters(
            structure_candidate.room_refiner
        ),
        "structure_refiner": _count_parameters(
            structure_candidate.structure_refiner
        ),
    }
    if structure_candidate.structure_refiner.global_route is not None:
        structure_candidate_groups["structure_mita_route"] = (
            _count_parameters(
                structure_candidate.structure_refiner.global_route
            )
        )
    opening_config = _read(args.opening_router_config)["model"]
    opening_candidate = OpeningHostRouter(
        int(opening_config["hidden_dim"]),
        num_types=len(taxonomy.opening_ids),
        route_attention=bool(opening_config["route_attention"]),
        route_router_count=int(opening_config["route_router_count"]),
        route_top_k=int(opening_config["route_top_k"]),
    )

    report = {
        "schema_version": "residential-model-budget-audit-v1",
        "status": "complete",
        "scope": "residential_cubicasa_only",
        "room_model": {
            "checkpoint": str(args.room_checkpoint),
            "parameters": room_total,
            "parameter_groups": room_groups,
            "parameter_share": {
                name.rstrip("."): value / max(1, room_total)
                for name, value in room_groups.items()
            },
            "observed_budget": _budget(5122, 8, 8),
            "assessment": (
                "Sufficient for relative mechanism selection; absolute room "
                "quality is readout-limited rather than encoder-limited."
            ),
            "candidate": {
                "config": str(args.room_candidate_config),
                "parameters": room_candidate_total,
                "parameter_groups": room_candidate_groups,
                "parameter_share": {
                    name: value / max(1, room_candidate_total)
                    for name, value in room_candidate_groups.items()
                },
                "screen_budget": _budget(512, 16, 8),
                "promotion_budget": _budget(1024, 20, 8),
                "confirmation_budget": _budget(5122, 20, 8),
            },
        },
        "predicted_channel_model": {
            "static": {
                "parameters": _count_parameters(static_model),
                "budget": _budget(256, 8, 4),
                "assessment": "underbudgeted_for_architecture_claims",
            },
            "dynamic": {
                "parameters": _count_parameters(dynamic_model),
                "screen_budget": _budget(512, 16, 4),
                "promotion_budget": _budget(1024, 20, 8),
                "confirmation_budget": _budget(5126, 20, 16),
                "assessment": "source_router_head_strengthened_budget",
            },
        },
        "structure_bridge": {
            "checkpoint": str(args.structure_checkpoint),
            "parameters": structure_params,
            "previous_full_budget": _budget(5126, 3, 16),
            "next_budget": _budget(5126, 12, 16),
            "assessment": (
                "The old full run was too short for a stable wall/opening "
                "claim; retrain only on the frozen room distribution."
            ),
            "candidate": {
                "config": str(args.structure_candidate_config),
                "parameters": _count_parameters(structure_candidate),
                "parameter_groups": structure_candidate_groups,
                "screen_budget": _budget(512, 12, 8),
                "promotion_budget": _budget(1024, 16, 8),
                "confirmation_budget": _budget(5126, 16, 16),
            },
        },
        "opening_host_router": {
            "config": str(args.opening_router_config),
            "parameters": _count_parameters(opening_candidate),
            "mita_route_parameters": (
                _count_parameters(opening_candidate.route)
                if opening_candidate.route is not None
                else 0
            ),
            "screen_budget": _budget(256, 16, 4),
            "promotion_budget": _budget(1024, 20, 4),
            "confirmation_budget": _budget(5126, 16, 8),
        },
        "historical_weak_metric_capacity_budget": balanced_config,
        "decision": {
            "increase_predicted_channel_capacity": True,
            "increase_predicted_channel_updates": True,
            "increase_room_encoder_capacity": False,
            "increase_room_readout_capacity": True,
            "add_sparse_stage_memory_for_weak_cad_heads": True,
            "expand_every_attention_module": False,
            "increase_structure_bridge_budget_after_protocol_alignment": True,
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
