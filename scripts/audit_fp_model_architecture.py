#!/usr/bin/env python3
"""Build a matched-budget architecture audit for FP completion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from floorflow.data.control_contract import sha256_file
from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.architecture_profile import (
    profile_callable,
    resource_dict,
)
from floorflow.evaluation.paired_bootstrap import paired_delta
from floorflow.heads.coarse_furniture_refiner import CoarseFurnitureRefiner
from floorflow.heads.coarse_layout_refiner import CoarseLayoutRefiner
from floorflow.heads.coarse_structure_refiner import CoarseStructureRefiner
from floorflow.heads.coarse_support_refiner import CoarseSupportRefiner

COMPONENT_DIRS = {
    "room": "no_room",
    "structure": "no_structure",
    "furniture": "no_furniture",
    "support": "no_support",
}

PRIMARY_METRICS = {
    "room": {
        "room_macro_iou": "higher",
        "room_adjacency_f1": "higher",
    },
    "structure": {
        "fpeval_wall_f1": "higher",
        "fpeval_opening_f1": "higher",
    },
    "furniture": {"furniture_iou": "higher"},
    "support": {"room_support_iou": "higher"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--table",
        type=Path,
        default=Path("paper/tables/fp_component_audit.md"),
    )
    parser.add_argument("--taxonomy", type=Path, default=Path("outputs/cache/cubicasa5k_full_v2/taxonomy.json"))
    parser.add_argument("--room-checkpoint", type=Path, default=Path("outputs/train/layout_refiner_full_s224/best.pt"))
    parser.add_argument(
        "--structure-checkpoint",
        type=Path,
        default=Path("outputs/train/coarse_structure_type_full_v1/best.pt"),
    )
    parser.add_argument(
        "--furniture-checkpoint",
        type=Path,
        default=Path("outputs/train/coarse_furniture_full_s192_v1/best.pt"),
    )
    parser.add_argument(
        "--support-checkpoint",
        type=Path,
        default=Path("outputs/train/coarse_support_full_v1/best.pt"),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument(
        "--component-ablation-root",
        type=Path,
        default=Path("outputs/eval/fp_architecture_ablation_val191_v1"),
    )
    parser.add_argument(
        "--multiseed-report",
        type=Path,
        default=Path("outputs/reports/fp_hard_metrics_multiseed_test503_v1.json"),
    )
    parser.add_argument(
        "--raw-comparison",
        type=Path,
        default=Path("outputs/reports/fp_hard_metrics_full_vs_raw_test503_v1.json"),
    )
    parser.add_argument(
        "--support-report",
        type=Path,
        default=Path("outputs/reports/fp_room_label_support_test503_v1.json"),
    )
    parser.add_argument(
        "--mita-report",
        type=Path,
        default=Path("outputs/reports/a1_mita_ablation.json"),
    )
    parser.add_argument(
        "--mita-mask-report",
        type=Path,
        default=Path("outputs/reports/c3_attention_ablation.json"),
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_module(
    name: str,
    checkpoint: Path,
    taxonomy: FloorplanTaxonomy,
    device: torch.device,
) -> torch.nn.Module:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if name == "room":
        module = CoarseLayoutRefiner(len(taxonomy.room_ids))
    elif name == "structure":
        module = CoarseStructureRefiner(
            len(taxonomy.room_ids) + len(taxonomy.opening_ids) + 3,
            len(taxonomy.opening_ids),
            base_channels=int(payload["config"].get("base_channels", 32)),
        )
    elif name == "furniture":
        module = CoarseFurnitureRefiner(
            len(taxonomy.room_ids) + len(taxonomy.furniture_ids) + 1,
            len(taxonomy.furniture_ids),
            base_channels=int(payload["config"].get("base_channels", 32)),
        )
    elif name == "support":
        module = CoarseSupportRefiner(
            len(taxonomy.room_ids) + 4,
            base_channels=int(payload["config"].get("base_channels", 32)),
        )
    else:
        raise ValueError(f"unknown component: {name}")
    module.load_state_dict(payload["model"])
    return module.to(device).eval()


def _profile_inputs(name: str, taxonomy: FloorplanTaxonomy, device: torch.device):
    batch = 1
    footprint = torch.ones(batch, 1, 256, 256, device=device)
    room_count = len(taxonomy.room_ids)
    if name == "room":
        coarse = torch.randn(batch, room_count, 224, 224, device=device)
        return lambda module: module(coarse, footprint)
    if name == "structure":
        control = torch.rand(
            batch,
            room_count + len(taxonomy.opening_ids) + 3,
            128,
            128,
            device=device,
        )
        return lambda module: module(control)
    if name == "furniture":
        control = torch.rand(
            batch,
            room_count + len(taxonomy.furniture_ids) + 1,
            192,
            192,
            device=device,
        )
        return lambda module: module(control, output_size=(256, 256))
    control = torch.rand(batch, room_count + 4, 192, 192, device=device)
    return lambda module: module(control, output_size=(256, 256))


def profile_components(args, taxonomy: FloorplanTaxonomy) -> dict[str, dict]:
    device = torch.device(args.device)
    checkpoints = {
        "room": args.room_checkpoint,
        "structure": args.structure_checkpoint,
        "furniture": args.furniture_checkpoint,
        "support": args.support_checkpoint,
    }
    result = {}
    for name, checkpoint in checkpoints.items():
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        module = _load_module(name, checkpoint, taxonomy, device)
        forward = _profile_inputs(name, taxonomy, device)
        profile = profile_callable(
            module,
            lambda module=module, forward=forward: forward(module),
            device=device,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        result[name] = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            **resource_dict(profile),
        }
    resource_names = ("parameters", "macs", "latency_ms_mean", "peak_memory_bytes")
    result["full_pipeline_sum"] = {
        name: (
            sum(
                int(result[component][name])
                for component in checkpoints
                if result[component][name] is not None
            )
            if all(result[component][name] is not None for component in checkpoints)
            else None
        )
        for name in resource_names
    }
    result["full_pipeline_sum"]["gmacs"] = result["full_pipeline_sum"]["macs"] / 1.0e9
    result["profile_contract"] = {
        "batch_size": 1,
        "resolution": 256,
        "device": str(device),
        "precision": "default",
        "warmup": args.warmup,
        "repeats": args.repeats,
        "aggregation": "serial sum; measured per-stage latency is not parallelized",
    }
    return result


def _screen_component(
    component: str,
    full_rows: list[dict],
    disabled_rows: list[dict],
) -> dict:
    comparisons = {
        metric: paired_delta(full_rows, disabled_rows, metric)
        for metric in PRIMARY_METRICS[component]
    }
    checks = {}
    for metric, direction in PRIMARY_METRICS[component].items():
        stat = comparisons[metric]
        if direction == "higher":
            checks[metric] = stat["ci_low"] > 0.0
        else:
            checks[metric] = stat["ci_high"] < 0.0
    return {
        "matched_control": f"disable {component}; all other checkpoints fixed",
        "comparisons_full_minus_disabled": comparisons,
        "checks": checks,
        "passed": all(checks.values()),
    }


def audit_component_ablations(root: Path) -> dict:
    full_path = root / "full" / "per_case.json"
    if not full_path.is_file():
        return {
            "status": "pending",
            "root": str(root),
            "reason": "matched component-off runs have not been generated",
        }
    full_rows = read_json(full_path)
    components = {}
    for component, directory in COMPONENT_DIRS.items():
        path = root / directory / "per_case.json"
        if not path.is_file():
            components[component] = {"status": "missing", "per_case": str(path)}
            continue
        components[component] = {
            "status": "complete",
            "per_case": str(path),
            **_screen_component(component, full_rows, read_json(path)),
        }
    return {
        "status": (
            "complete"
            if all(item.get("status") == "complete" for item in components.values())
            else "partial"
        ),
        "root": str(root),
        "case_count": len(full_rows),
        "components": components,
    }


def summarize_existing_failure_modes(args) -> dict:
    multiseed = read_json(args.multiseed_report)
    raw = read_json(args.raw_comparison)
    support = read_json(args.support_report)
    mita = read_json(args.mita_report)
    mita_mask = read_json(args.mita_mask_report)
    support_means = [
        row["metrics"]["room_support_iou"] for row in multiseed["seeds"]
    ]
    room_label_support_means = [row["mean"] for row in support["seeds"]]
    return {
        "clean_complete_control": {
            "summary": multiseed["summary"],
            "raw_control_comparisons": raw["paired"],
            "checks": multiseed["checks"],
        },
        "support_refiner": {
            "seed_support_iou": support_means,
            "seed_range": max(support_means) - min(support_means),
            "room_label_support_iou": room_label_support_means,
            "room_label_support_range": (
                max(room_label_support_means) - min(room_label_support_means)
            ),
            "decision": (
                "replace with room-label support"
                if max(support_means) - min(support_means) > 0.05
                else "stabilized"
            ),
        },
        "mita_graph_adapter": {
            "delta_minus_mha": mita["delta_mita_minus_mha"],
            "masking_conclusion": mita_mask["finding"],
            "not_worse_all_mask_ratios": mita_mask[
                "mita_not_worse_all_ratios"
            ],
            "decision": (
                "exclude from main completion stack"
                if not mita_mask["mita_not_worse_all_ratios"]
                else "retain for topology ablation"
            ),
        },
    }


def finalize_decisions(ablations: dict, failure_modes: dict) -> dict:
    decisions = {}
    components = ablations.get("components", {})
    for component in COMPONENT_DIRS:
        screen = components.get(component, {})
        if screen.get("status") == "complete":
            decisions[component] = (
                "retain" if screen["passed"] else "remove_or_replace"
            )
        else:
            decisions[component] = "unresolved"
    decisions["support"] = failure_modes["support_refiner"]["decision"]
    decisions["mita"] = failure_modes["mita_graph_adapter"]["decision"]
    return decisions


def write_table(path: Path, report: dict) -> None:
    rows = [
        "# FP Component Audit",
        "",
        "| Component | Parameters | GMACs | Latency (ms) | Decision | Matched evidence |",
        "|---|---:|---:|---:|---|---|",
    ]
    for component in ("room", "structure", "furniture", "support"):
        resource = report["profiles"][component]
        screen = report["component_ablations"]["components"][component]
        comparisons = screen["comparisons_full_minus_disabled"]
        evidence = "; ".join(
            f"{metric} {stats['mean_delta']:+.4f} [{stats['ci_low']:+.4f}, {stats['ci_high']:+.4f}]"
            for metric, stats in comparisons.items()
        )
        rows.append(
            f"| {component} | {resource['parameters']:,} | "
            f"{resource['gmacs']:.2f} | {resource['latency_ms_mean']:.2f} | "
            f"{report['decisions'][component]} | {evidence} |"
        )
    rows.extend(
        [
            "",
            (
                "Intervals are paired case-level bootstrap intervals for full "
                "minus the same-checkpoint component-off path."
            ),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    taxonomy = FloorplanTaxonomy.read(args.taxonomy)
    profiles = profile_components(args, taxonomy)
    ablations = audit_component_ablations(args.component_ablation_root)
    failure_modes = summarize_existing_failure_modes(args)
    decisions = finalize_decisions(ablations, failure_modes)
    retained = sorted(
        name
        for name, decision in decisions.items()
        if decision in {"retain", "stabilized"}
    )
    report = {
        "status": (
            "complete"
            if ablations["status"] == "complete"
            else "resource_profile_complete_ablation_pending"
        ),
        "scope": "matched-budget audit of the complete-control FP pipeline",
        "best_of_n_used": False,
        "profiles": profiles,
        "component_ablations": ablations,
        "failure_mode_evidence": failure_modes,
        "decisions": decisions,
        "retained_modules": retained,
        "promotion_rule": (
            "A module is retained only when the full model beats the same-checkpoint, "
            "same-control component-off path on every predeclared metric with a paired "
            "bootstrap interval excluding zero."
        ),
        "claim_implication": (
            "Only retained_modules may appear in the main model claim; removed modules "
            "remain diagnostic analyses or are replaced by their matched control."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_table(args.table, report)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "table": str(args.table),
                "status": report["status"],
                "decisions": decisions,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
