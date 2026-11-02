#!/usr/bin/env python3
"""Audit the same-parameter architecture gap to the pinned Trinity d192 model."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party/Trinity/src"))

from trinity.models.backbone import SetTransformerDenoiser
from trinity.models.presets import DenoiserArchConfig

from floorflow.chip_flow.flow_adapter import build_chip_model
from floorflow.chip_flow.slot_model import ChipSlotFlowModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--current-checkpoint",
        type=Path,
        default=ROOT
        / "outputs/chip_final/screen_20k/graph_velocity_seed19_20k_final/checkpoints/final.pt",
    )
    parser.add_argument(
        "--trinity-config",
        type=Path,
        default=ROOT / "third_party/Trinity/checkpoints/d192/config.json",
    )
    parser.add_argument(
        "--trinity-report",
        type=Path,
        default=ROOT / "outputs/reports/trinity_d192_matched_protocol_v2.json",
    )
    parser.add_argument(
        "--current-report",
        type=Path,
        default=ROOT / "outputs/reports/chip_final_three_seed_evidence_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_trinity_parity_gap_v1.json",
    )
    return parser.parse_args()


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def first_party_python_sizes() -> list[dict[str, int | str]]:
    paths = [
        path
        for root in ("src", "tests", "scripts")
        for path in (ROOT / root).rglob("*.py")
    ]
    return sorted(
        (
            {"path": str(path.relative_to(ROOT)), "lines": len(path.read_text().splitlines())}
            for path in paths
        ),
        key=lambda row: int(row["lines"]),
        reverse=True,
    )


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(
        args.current_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    current_config = dict(checkpoint["config"])
    model_arguments = inspect.signature(ChipSlotFlowModel).parameters
    current = build_chip_model(
        {
            key: value
            for key, value in current_config.items()
            if key == "architecture" or key in model_arguments
        }
    )
    trinity_config = json.loads(args.trinity_config.read_text(encoding="utf-8"))
    trinity = SetTransformerDenoiser(
        DenoiserArchConfig(**trinity_config["arch"])
    )
    trinity_report = json.loads(args.trinity_report.read_text(encoding="utf-8"))
    current_report = json.loads(args.current_report.read_text(encoding="utf-8"))

    current_parameters = parameter_count(current)
    trinity_parameters = parameter_count(trinity)
    parameter_gap = abs(current_parameters - trinity_parameters) / trinity_parameters
    current_refined = current_report["seed_mean_absolute"]["refined"]["graph"]
    trinity_cost = float(trinity_report["mean_cost"])
    trinity_v_rel = float(trinity_report["mean_v_rel"])
    current_cost = float(current_refined["legalized_hard_cost"])
    current_v_rel = float(current_refined["legalized_v_rel"])
    python_sizes = first_party_python_sizes()
    max_python = python_sizes[0]
    reuse_sources = [
        "third_party/Trinity/src/trinity/models/backbone.py",
        "third_party/Trinity/src/trinity/models/block.py",
        "third_party/Trinity/src/trinity/models/components/attention.py",
        "third_party/Trinity/src/trinity/models/components/embed.py",
        "third_party/Trinity/src/trinity/models/components/time_cond.py",
        "third_party/Trinity/src/trinity/models/components/graph_mix.py",
        "third_party/Trinity/src/trinity/conditioning/graph_pe.py",
    ]
    checks = {
        "same_parameter_budget_within_1pct": parameter_gap <= 0.01,
        "same_feature_width": int(current_config["feature_dim"]) == int(
            trinity_config["arch"]["feature_dim"]
        ),
        "matched_metrics_available": all(
            value > 0 for value in (current_cost, current_v_rel, trinity_cost, trinity_v_rel)
        ),
        "reuse_sources_exist": all(
            (ROOT / path).is_file() for path in reuse_sources
        ),
        "first_party_python_below_600": int(max_python["lines"]) < 600,
    }
    report = {
        "status": "pass" if all(checks.values()) else "repair_audit_inputs",
        "scope": "same-protocol architecture parity audit; no training or quality claim",
        "current": {
            "checkpoint": str(args.current_checkpoint),
            "architecture": current_config["architecture"],
            "parameters": current_parameters,
            "feature_dim": int(current_config["feature_dim"]),
            "flow_state": "one invariant global vector per case",
            "layers": {
                "condition": int(current_config["condition_layers"]),
                "target": int(current_config["target_layers"]),
                "decoder": int(current_config["decoder_layers"]),
            },
            "block_features": "19-column graph conditioning",
            "attention": "explicit QKV graph bias",
            "norm_mlp": "LayerNorm + SiLU MLP",
            "time_conditioning": "projected scalar time added to global velocity input",
            "graph_pe": "none",
            "refined_cost": current_cost,
            "refined_v_rel": current_v_rel,
        },
        "trinity": {
            "config": str(args.trinity_config),
            "parameters": trinity_parameters,
            "feature_dim": int(trinity_config["arch"]["feature_dim"]),
            "flow_state": "one latent vector per block",
            "layers": int(trinity_config["arch"]["depth"]),
            "hidden": int(trinity_config["arch"]["hidden"]),
            "heads": int(trinity_config["arch"]["heads"]),
            "head_dim": int(trinity_config["arch"]["head_dim"]),
            "block_features": "19-column semantics with normalized pin pull; soft-MIB group mean projection is not yet bridged",
            "attention": trinity_config["arch"]["attn"],
            "norm_mlp": (
                f"{trinity_config['arch']['norm']} + "
                f"{trinity_config['arch']['mlp']}"
            ),
            "time_conditioning": trinity_config["arch"]["time_cond"],
            "graph_pe": trinity_config["graph_pe"],
            "refined_cost": trinity_cost,
            "refined_v_rel": trinity_v_rel,
        },
        "gap": {
            "parameter_relative": parameter_gap,
            "cost_absolute": current_cost - trinity_cost,
            "cost_relative": current_cost / trinity_cost - 1.0,
            "v_rel_absolute": current_v_rel - trinity_v_rel,
            "v_rel_relative": current_v_rel / trinity_v_rel - 1.0,
        },
        "architecture_gaps": [
            "The current flow transports one global vector while Trinity transports every block token.",
            "Trinity uses two spectral graph-drawing coordinates at the input; the current model has no graph PE.",
            "The current 12-ish block budget is split into separate 2/4/4 encoders and decoder rather than a single 12-layer modulated token transformer.",
            "Trinity uses RMSNorm, SwiGLU, qk-normalized SDPA, and token time conditioning; the current model uses LayerNorm, SiLU, explicit QKV, and scalar time injection.",
            "The current decoder injects global state but the state is nearly case-invariant and causally weak.",
            "The first token bridge preserves tensor anchors but does not yet port Trinity's all-soft MIB group-mean sampling projection.",
        ],
        "ordered_interventions": [
            "Wrap the pinned SetTransformerDenoiser in the existing chip adapter without copying its internals.",
            "Build dense adjacency and spectral_draw(k=2) from the existing cache, preserving feature and mask semantics.",
            "Use per-block latent states and x0 prediction while retaining exact-area decode and current physics terms.",
            "Run a same-parameter CPU forward/backward/permutation/padding gate before any GPU training.",
            "Only after a 5k matched win, test an optional invariant-global residual as the paper's architecture delta.",
        ],
        "reuse_sources": reuse_sources,
        "engineering": {
            "max_first_party_python": max_python,
            "first_party_python_under_600": bool(checks["first_party_python_below_600"]),
        },
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
