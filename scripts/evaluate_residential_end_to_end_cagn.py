"""Evaluate hard-masked CAGN action selection on real model candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import torch

from floorflow.data.constraint_action_dataset import ACTION_NAMES
from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.paired_bootstrap import paired_delta
from floorflow.inference.constraint_action_runtime import (
    ActionCandidate,
    ConstraintActionRuntime,
    hard_rule_selection,
)
from floorflow.model.constraint_aware_gate import ConstraintAwareGate

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "outputs/control_contracts/model_candidates_v1/val/index.json"
FREEZE = ROOT / "outputs/reports/residential_cagn_validation_freeze_v1.json"
CHECKPOINT = ROOT / "outputs/residential_cagn_gate_v3/seed19.pt"
OUTPUT = ROOT / "outputs/reports/residential_cagn_runtime_val_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bootstrap", type=int, default=4000)
    return parser.parse_args()


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _candidates(group: dict) -> list[ActionCandidate]:
    rows = []
    operation_id = ACTION_NAMES.index(group["operation"])
    for row in group["candidates"]:
        rows.append(
            ActionCandidate(
                candidate_id=row["candidate_id"],
                role=row["role"],
                node_features=torch.tensor(row["node_features"]),
                adjacency=torch.tensor(row["adjacency"]),
                global_features=torch.tensor(row["global_features"]),
                constraint_diagnostics=torch.tensor(
                    row["constraint_diagnostics"]
                ),
                action_features=torch.tensor(row["action_features"]),
                hard_mask=torch.tensor(
                    list(row["hard_mask"].values()),
                    dtype=torch.bool,
                ),
                declared_legal=bool(row["hard_feasible"]),
                operation_id=operation_id,
                utility=float(row["quality"]["utility"]),
            )
        )
    return rows


def _summary(
    groups: list[dict],
    decisions: list,
    *,
    label: str,
) -> tuple[dict, list[dict]]:
    components = defaultdict(list)
    legal = []
    invalid_mass = []
    hard_feasible = []
    case_rows = defaultdict(list)
    for group, decision in zip(groups, decisions, strict=True):
        candidate = group["candidates"][decision.index]
        quality = candidate["quality"]
        case_rows[group["name"]].append(float(quality["utility"]))
        for key in (
            "utility",
            "room_intent_iou",
            "wall_intent_iou",
            "opening_intent_iou",
            "furniture_intent_iou",
            "room_mse",
            "target_room_iou",
        ):
            value = quality[key] if key != "utility" else quality["utility"]
            components[key].append(float(value))
        legal.append(candidate["role"] != "near_violation")
        invalid_mass.append(decision.invalid_probability_mass)
        hard_feasible.append(decision.hard_feasible)
    rows = [
        {"name": name, "utility": sum(values) / len(values)}
        for name, values in sorted(case_rows.items())
    ]
    return {
        "method": label,
        "groups": len(groups),
        "cases": len(case_rows),
        "selected_legal_rate": sum(legal) / max(len(legal), 1),
        "hard_feasible_rate": sum(hard_feasible) / max(len(hard_feasible), 1),
        "invalid_probability_mass": sum(invalid_mass),
        "metrics": {
            key: sum(values) / max(len(values), 1)
            for key, values in components.items()
        },
    }, rows


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    taxonomy = FloorplanTaxonomy.read(
        ROOT / "outputs/cache/cubicasa5k_full_v2/taxonomy.json"
    )
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    selected = freeze["selected"]
    if _hash(CHECKPOINT) != selected["checkpoint_sha256"]:
        raise ValueError("runtime checkpoint does not match validation freeze")
    payload = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=False,
    )
    model = ConstraintAwareGate(**payload["config"]).to(device)
    model.load_state_dict(payload["model"])
    runtime = ConstraintActionRuntime(
        model,
        alpha=float(selected["alpha"]),
        legal_fallback_threshold=float(
            selected["legal_fallback_threshold"]
        ),
    )
    groups = json.loads(INDEX.read_text(encoding="utf-8"))["groups"]
    cagn_decisions = []
    hard_decisions = []
    for group in groups:
        candidates = _candidates(group)
        hard_decisions.append(hard_rule_selection(candidates))
        cagn_decisions.append(
            runtime.select(candidates, taxonomy, device=device)
        )
    hard_summary, hard_rows = _summary(
        groups,
        hard_decisions,
        label="hard_rule_hash",
    )
    cagn_summary, cagn_rows = _summary(
        groups,
        cagn_decisions,
        label="cagn_runtime",
    )
    comparison = paired_delta(
        cagn_rows,
        hard_rows,
        "utility",
        iterations=args.bootstrap,
        seed=6101,
    )
    mechanical_checks = {
        "single_decision_per_group": (
            len(cagn_decisions) == len(groups)
            and len(hard_decisions) == len(groups)
        ),
        "hard_feasible_selection": cagn_summary["hard_feasible_rate"] == 1.0,
        "invalid_probability_mass_zero": (
            cagn_summary["invalid_probability_mass"] == 0.0
        ),
    }
    quality_checks = {
        "selected_legal_rate": cagn_summary["selected_legal_rate"] == 1.0,
        "utility_improves_over_hard_gate": (
            comparison["mean_delta"] > 0.0 and comparison["ci_low"] > 0.0
        ),
    }
    report = {
        "schema_version": "residential-cagn-runtime-val-v1",
        "status": "passed" if all(mechanical_checks.values()) else "failed",
        "quality_status": (
            "passed" if all(quality_checks.values()) else "pending_training"
        ),
        "scope": "residential_floorplan_only_validation",
        "candidate_origin": "real predicted-control model candidates",
        "single_trajectory": True,
        "test_best_of_n": False,
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "checkpoint_sha256": selected["checkpoint_sha256"],
        "hard_gate": hard_summary,
        "cagn_runtime": cagn_summary,
        "paired_utility": comparison,
        "mechanical_checks": mechanical_checks,
        "quality_checks": quality_checks,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(OUTPUT),
                "mechanical_checks": mechanical_checks,
                "quality_checks": quality_checks,
            },
            indent=2,
        )
    )
    if not all(mechanical_checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
