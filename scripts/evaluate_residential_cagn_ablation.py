"""Evaluate five matched CAGN arms on the held-out validation split."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from floorflow.evaluation.constraint_ranking import ndcg_at_k
from floorflow.evaluation.paired_bootstrap import paired_delta
from floorflow.model.constraint_aware_gate import ConstraintAwareGate

ROOT = Path(__file__).resolve().parents[1]
GROUP_KEYS = (
    "node_features",
    "adjacency",
    "global_features",
    "constraint_diagnostics",
    "node_mask",
    "operation_ids",
    "action_features",
    "hard_mask",
    "legal",
    "violation_labels",
    "candidate_hash",
    "group_index",
    "room_count",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature-root",
        type=Path,
        default=ROOT / "outputs/control_contracts/residential_cagn_features_v1",
    )
    parser.add_argument(
        "--hard-learnable-checkpoint",
        type=Path,
        default=ROOT / "outputs/residential_cagn_gate_scratch_v3/seed20.pt",
    )
    parser.add_argument(
        "--full-checkpoint",
        type=Path,
        default=ROOT / "outputs/residential_cagn_gate_v3/seed20.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/residential_cagn_five_arm_v1.json",
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=ROOT / "paper/tables/residential_cagn_ablation_v1.md",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=19)
    return parser.parse_args()


def _load_batch(path: Path, device: torch.device) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    features = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in payload.items()
    }
    groups = defaultdict(list)
    for row, group in enumerate(features["group_index"].tolist()):
        groups[int(group)].append(row)
    sizes = {len(rows) for rows in groups.values()}
    if len(sizes) != 1:
        raise ValueError(f"unequal candidate groups: {sizes}")
    indices = torch.tensor(
        list(groups.values()),
        device=features["group_index"].device,
    )
    return {
        key: features[key][indices]
        for key in GROUP_KEYS
    } | {
        "family_ids": features["family_ids"],
        "groups": len(groups),
    }


def _graph_inputs(batch: dict) -> tuple[torch.Tensor, ...]:
    groups, candidates = batch["node_features"].shape[:2]
    return (
        batch["node_features"].flatten(0, 1),
        batch["adjacency"].flatten(0, 1),
        batch["family_ids"][None].expand(groups * candidates, -1),
        batch["operation_ids"].flatten(),
        batch["action_features"].flatten(0, 1),
        batch["hard_mask"].flatten(0, 1),
        batch["node_mask"].flatten(0, 1),
        torch.cat(
            [
                batch["global_features"].flatten(0, 1),
                batch["constraint_diagnostics"].flatten(0, 1),
            ],
            dim=-1,
        ),
    )


@torch.no_grad()
def _model_scores(
    model: ConstraintAwareGate,
    batch: dict,
) -> tuple[np.ndarray, np.ndarray, bool]:
    model.eval()
    output = model(*_graph_inputs(batch))
    shape = batch["legal"].shape
    hard = output["hard_feasible"].reshape(shape)
    logits = output["gate_logits"].reshape(shape)
    probability = output["gate_probability"].reshape(shape)
    invalid_zero = bool(
        torch.equal(
            probability[~hard],
            torch.zeros_like(probability[~hard]),
        )
    )
    return logits.cpu().numpy(), hard.cpu().numpy(), invalid_zero


def _hard_score(batch: dict) -> np.ndarray:
    hard = batch["hard_mask"].all(dim=-1).cpu().numpy()
    hashes = batch["candidate_hash"].cpu().numpy().astype(np.float64)
    hashes = hashes / np.maximum(1.0, hashes.max(axis=1, keepdims=True))
    return np.where(hard, 1000.0 + hashes, -np.inf)


def _baseline_score(batch: dict) -> np.ndarray:
    hashes = batch["candidate_hash"].cpu().numpy().astype(np.float64)
    return hashes / np.maximum(1.0, hashes.max(axis=1, keepdims=True))


def _rows(
    scores: np.ndarray,
    batch: dict,
) -> tuple[dict, list[dict]]:
    legal = batch["legal"].cpu().numpy()
    violation = batch["violation_labels"].any(dim=-1).cpu().numpy()
    operations = batch["operation_ids"][:, 0].cpu().numpy()
    ndcgs = []
    selected_legal = []
    selected_violation = []
    per_group = []
    operation_values = defaultdict(list)
    for index in range(scores.shape[0]):
        score = float(ndcg_at_k(scores[index], legal[index].astype(float)))
        chosen = int(np.argmax(scores[index]))
        row = {
            "ndcg_at_5": score,
            "selected_legal": float(bool(legal[index, chosen])),
            "selected_violation": float(bool(violation[index, chosen])),
            "operation": int(operations[index]),
        }
        per_group.append(row)
        ndcgs.append(score)
        selected_legal.append(row["selected_legal"])
        selected_violation.append(row["selected_violation"])
        operation_values[str(int(operations[index]))].append(score)
    return {
        "groups": len(ndcgs),
        "ndcg_at_5": float(np.mean(ndcgs)),
        "selected_legal_rate": float(np.mean(selected_legal)),
        "selected_violation_rate": float(np.mean(selected_violation)),
        "operation_ndcg": {
            key: float(np.mean(values))
            for key, values in operation_values.items()
        },
    }, per_group


def _load_model(path: Path, device: torch.device) -> ConstraintAwareGate:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = ConstraintAwareGate(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    batch = _load_batch(args.feature_root / "val/features.pt", device)
    arms = {}
    group_rows = {}

    def add_arm(name, scores, invalid_zero=True):
        summary, rows = _rows(scores, batch)
        summary["invalid_probability_exact_zero"] = invalid_zero
        arms[name] = summary
        group_rows[name] = rows

    add_arm("baseline", _baseline_score(batch))
    add_arm("hard_gate", _hard_score(batch))
    hard_model = _load_model(args.hard_learnable_checkpoint, device)
    scores, hard, exact = _model_scores(hard_model, batch)
    scores = np.where(hard, scores, -np.inf)
    add_arm("hard_plus_learnable", scores, exact)
    arms["hard_plus_learnable"]["parameters"] = sum(
        parameter.numel() for parameter in hard_model.parameters()
    )
    full_model = _load_model(args.full_checkpoint, device)
    scores, hard, exact = _model_scores(full_model, batch)
    scores = np.where(hard, scores, -np.inf)
    add_arm("full_constraint_embedding", scores, exact)
    arms["full_constraint_embedding"]["parameters"] = sum(
        parameter.numel() for parameter in full_model.parameters()
    )
    random_model = ConstraintAwareGate().to(device)
    scores, hard, exact = _model_scores(random_model, batch)
    add_arm("parameter_matched_random_gate", np.where(hard, scores, -np.inf), exact)
    arms["parameter_matched_random_gate"]["parameters"] = sum(
        parameter.numel() for parameter in random_model.parameters()
    )
    comparisons = {
        "full_minus_hard": paired_delta(
            group_rows["full_constraint_embedding"],
            group_rows["hard_gate"],
            "ndcg_at_5",
            iterations=args.bootstrap,
            seed=args.seed,
        ),
        "full_minus_learnable": paired_delta(
            group_rows["full_constraint_embedding"],
            group_rows["hard_plus_learnable"],
            "ndcg_at_5",
            iterations=args.bootstrap,
            seed=args.seed + 1,
        ),
        "full_minus_random": paired_delta(
            group_rows["full_constraint_embedding"],
            group_rows["parameter_matched_random_gate"],
            "ndcg_at_5",
            iterations=args.bootstrap,
            seed=args.seed + 2,
        ),
    }
    checks = {
        "all_arms_present": len(arms) == 5,
        "full_beats_hard_ndcg": arms["full_constraint_embedding"]["ndcg_at_5"] > arms["hard_gate"]["ndcg_at_5"],
        "full_ndcg_ci_excludes_zero": comparisons["full_minus_hard"]["ci_low"] > 0.0,
        "full_beats_random": arms["full_constraint_embedding"]["ndcg_at_5"] > arms["parameter_matched_random_gate"]["ndcg_at_5"],
        "hard_mask_exact": all(
            arm.get("invalid_probability_exact_zero", True)
            for arm in arms.values()
        ),
        "parameter_matched": (
            arms["full_constraint_embedding"]["parameters"]
            == arms["parameter_matched_random_gate"]["parameters"]
        ),
    }
    report = {
        "schema_version": "residential-cagn-five-arm-v1",
        "status": "passed" if all(checks.values()) else "failed",
        "scope": "residential_floorplan_only_validation",
        "arms": arms,
        "paired_comparisons": comparisons,
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    table = [
        "# Residential CAGN five-arm validation",
        "",
        "| Arm | Params | NDCG@5 | Selected legal | Selected violation | Invalid mass |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, arm in arms.items():
        table.append(
            f"| {name} | {arm.get('parameters', 0):,} | "
            f"{arm['ndcg_at_5']:.4f} | {arm['selected_legal_rate']:.4f} | "
            f"{arm['selected_violation_rate']:.4f} | "
            f"{0.0 if arm['invalid_probability_exact_zero'] else 1.0:.1f} |"
        )
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text("\n".join(table) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "checks": checks}, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
