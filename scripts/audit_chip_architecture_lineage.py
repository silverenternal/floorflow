#!/usr/bin/env python3
"""Freeze the active chip lineage and recompute budget/evidence facts on CPU."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from floorflow.chip_flow.training_runtime import write_json_atomic
from floorflow.evaluation.paired_bootstrap import paired_delta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            ROOT
            / "outputs/chip_hierarchy/m27_training_seed_replication_v1"
            / "m27_seed21_5k/checkpoints/final.pt"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_architecture_lineage_audit_v1.json",
    )
    return parser.parse_args()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _checkpoint_facts(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    state = checkpoint["ema"]
    parameters = sum(value.numel() for value in state.values())
    steps = int(config["steps"])
    micro_batch = int(config["batch_size"])
    accumulation = int(config.get("grad_accum_steps", 1))
    return {
        "path": str(path.relative_to(ROOT)),
        "parameters": parameters,
        "training_exposures": steps * micro_batch * accumulation,
        "architecture": config["architecture"],
        "backbone_initialization": config.get("trinity_backbone_init"),
        "enabled_modules": {
            key: config.get(key)
            for key in (
                "relation_specific",
                "structural_relations",
                "coarse_commit_fraction",
                "dynamic_graph_relay",
                "dynamic_graph_input",
                "coarse_to_fine_conditioning",
                "structured_heads",
                "spatial_occupancy_conditioning",
                "spatial_route_field",
                "state_conditioned_edge_bias",
                "hierarchy_source_mode",
                "hierarchy_source_levels",
            )
        },
        "sampling_steps": int(config.get("sampling_steps", 0)),
        "optimizer": {
            "lr": config.get("lr"),
            "weight_decay": config.get("weight_decay"),
            "ema_decay": config.get("ema_decay"),
        },
    }


def _anchor_pairs() -> dict:
    portfolios = {
        19: "outputs/reports/chip_m48_portfolio_official100_postfix_seed19.json",
        20: "outputs/reports/chip_m48_portfolio_official100_seed20.json",
        21: "outputs/reports/chip_m48_portfolio_official100_seed21.json",
    }
    anchors = {
        19: "outputs/reports/chip_trinity_d192_nfe1_ref1600_official100_v1.json",
        20: "outputs/reports/chip_trinity_d192_nfe1_ref1600_official100_seed20.json",
        21: "outputs/reports/chip_trinity_d192_nfe1_ref1600_official100_seed21.json",
    }
    metrics = (
        "cost",
        "v_rel",
        "b2b_hpwl_gap",
        "p2b_hpwl_gap",
        "area_gap",
        "rho_mae",
        "grouping",
        "boundary",
    )
    current = {metric: [] for metric in metrics}
    reference = {metric: [] for metric in metrics}
    budget_checks = {}
    for seed, portfolio_path in portfolios.items():
        portfolio = _load(ROOT / portfolio_path)
        anchor = _load(ROOT / anchors[seed])
        budget_checks[str(seed)] = {
            "candidate": portfolio["budget"],
            "anchor": {
                "nfe": anchor["protocol"]["sampler_steps"],
                "refiner_steps": anchor["protocol"]["refiner"]["steps"],
                "legalizer_calls": 1,
            },
        }
        portfolio_rows = {int(row["index"]): row for row in portfolio["rows"]}
        anchor_rows = {int(row["index"]): row for row in anchor["rows"]}
        for index in sorted(set(portfolio_rows) & set(anchor_rows)):
            current_row = portfolio_rows[index]["selected"]["stages"]["legalized"]
            anchor_row = anchor_rows[index]
            key = seed * 1000 + index
            for metric in metrics:
                if metric == "cost":
                    candidate = current_row["cost"]
                    baseline = anchor_row.get(
                        "legalized_hard_cost",
                        anchor_row.get("cost"),
                    )
                elif metric == "v_rel":
                    candidate = current_row["v_rel"]
                    baseline = anchor_row.get(
                        "legalized_v_rel",
                        anchor_row.get("v_rel"),
                    )
                else:
                    candidate = current_row[metric]
                    baseline = anchor_row.get(f"legalized_{metric}")
                if baseline is None:
                    continue
                current[metric].append(
                    {"index": key, "value": float(candidate)}
                )
                reference[metric].append(
                    {"index": key, "value": float(baseline)}
                )
    paired = {}
    for metric in metrics:
        if not current[metric]:
            continue
        result = paired_delta(
            current[metric],
            reference[metric],
            "value",
            iterations=10000,
            lower_is_better=True,
        )
        result["candidate_mean"] = (
            result["baseline_mean"] + result["mean_delta"]
        )
        paired[metric] = result
    return {
        "scope": "post-fix three-portfolio join; repeated inference seeds are not independent",
        "candidate_budget": "2x NFE, 2x refiner, 3 legalizer calls",
        "anchor_budget": "1x NFE, 1x refiner, default legalizer",
        "budget_checks": budget_checks,
        "matched_budget_anchor_found": all(
            check["candidate"]["nfe"] == check["anchor"]["nfe"]
            and check["candidate"]["refiner_steps"]
            == check["anchor"]["refiner_steps"]
            and check["candidate"]["legalizer_calls"]
            == check["anchor"]["legalizer_calls"]
            for check in budget_checks.values()
        ),
        "paired": paired,
    }


def main() -> None:
    args = parse_args()
    checkpoint = _checkpoint_facts(args.checkpoint)
    trinity_parameters = 5_856_400
    trinity_exposures = 51_200_000
    report = {
        "version": "chip-architecture-lineage-audit-v1",
        "status": "canonical_lineage_frozen_with_open_stability_gate",
        "canonical_lineage": "typed_hierarchy_m27_trinity_backbone_init",
        "superseded_lineages": [
            "parameter_matched_per_block_graph_token_flow",
            "global_slot_control",
        ],
        "checkpoint": checkpoint,
        "parameter_budget": {
            "candidate": checkpoint["parameters"],
            "trinity_d192": trinity_parameters,
            "candidate_relative": checkpoint["parameters"] / trinity_parameters - 1.0,
        },
        "training_budget": {
            "candidate_exposures": checkpoint["training_exposures"],
            "trinity_d192_nominal": trinity_exposures,
            "anchor_over_candidate": (
                trinity_exposures / checkpoint["training_exposures"]
            ),
            "candidate_initialized_from_trinity": checkpoint[
                "backbone_initialization"
            ],
        },
        "stale_claims_to_remove": [
            "candidate is parameter-matched to Trinity d192",
            "candidate uses exactly 20x fewer nominal exposures",
            "M27 and graph-token candidate are interchangeable evidence",
            "post-fix seed20/seed21 portfolio evidence is missing",
        ],
        "postfix_portfolio_vs_anchor": _anchor_pairs(),
        "claim_decision": {
            "same_inference_quality": "training-seed stability unresolved",
            "absolute_budget_quality": "portfolio mean attractive but anchor must be rerun at equal 2x budget",
            "external_ppa": "not supported",
        },
    }
    write_json_atomic(
        args.output,
        json.loads(json.dumps(report, default=str)),
    )
    print(args.output)


if __name__ == "__main__":
    main()
