"""Stress-test CAGN ranking, hard masking, and invariance on validation."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from floorflow.evaluation.constraint_ranking import ndcg_at_k
from floorflow.model.constraint_aware_gate import ConstraintAwareGate
from floorflow.model.constraint_state_encoder import (
    d4_transform_graph,
    permute_graph_nodes,
)

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
        "--checkpoint",
        type=Path,
        default=ROOT / "outputs/residential_cagn_gate_v3/seed20.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/residential_cagn_stress_v1.json",
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=ROOT / "paper/tables/residential_cagn_stress_v1.md",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--seed", type=int, default=919)
    return parser.parse_args()


def _load_features(path: Path, device: torch.device) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in payload.items()
    }


def _group_batch(features: dict) -> dict:
    groups = defaultdict(list)
    for row, group in enumerate(features["group_index"].tolist()):
        groups[int(group)].append(row)
    sizes = {len(rows) for rows in groups.values()}
    if len(sizes) != 1:
        raise ValueError(f"unequal candidate group sizes: {sizes}")
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
        "candidates": len(next(iter(groups.values()))),
    }


def _graph_inputs(batch: dict) -> tuple[torch.Tensor, ...]:
    groups, candidates = batch["node_features"].shape[:2]
    family = batch["family_ids"]
    if family.ndim == 1:
        family = family[None].expand(groups * candidates, -1)
    elif family.ndim == 3:
        family = family.flatten(0, 1)
    return (
        batch["node_features"].flatten(0, 1),
        batch["adjacency"].flatten(0, 1),
        family,
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
def _score(model: ConstraintAwareGate, batch: dict) -> tuple[torch.Tensor, ...]:
    model.eval()
    output = model(*_graph_inputs(batch))
    shape = batch["legal"].shape
    return (
        output["gate_logits"].reshape(shape),
        output["gate_probability"].reshape(shape),
        output["hard_feasible"].reshape(shape),
    )


def _permuted(batch: dict, generator: torch.Generator) -> dict:
    groups, candidates, nodes, _ = batch["node_features"].shape
    nodes_flat = batch["node_features"].flatten(0, 1)
    adjacency = batch["adjacency"].flatten(0, 1)
    family = batch["family_ids"][None].expand(groups * candidates, -1)
    mask = batch["node_mask"].flatten(0, 1)
    out_nodes = []
    out_adjacency = []
    out_family = []
    out_mask = []
    for index in range(groups * candidates):
        permutation = torch.randperm(nodes, generator=generator).to(nodes_flat.device)
        node, edge = permute_graph_nodes(
            nodes_flat[index],
            adjacency[index],
            permutation,
        )
        out_nodes.append(node)
        out_adjacency.append(edge)
        out_family.append(family[index, permutation])
        out_mask.append(mask[index, permutation])
    return {
        **batch,
        "node_features": torch.stack(out_nodes).reshape(groups, candidates, nodes, -1),
        "adjacency": torch.stack(out_adjacency).reshape(
            groups,
            candidates,
            nodes,
            nodes,
        ),
        "family_ids": torch.stack(out_family).reshape(groups, candidates, nodes),
        "node_mask": torch.stack(out_mask).reshape(groups, candidates, nodes),
    }


def _rotated(batch: dict, transform: int) -> dict:
    nodes = batch["node_features"].clone()
    for group in range(nodes.shape[0]):
        for candidate in range(nodes.shape[1]):
            nodes[group, candidate] = d4_transform_graph(
                nodes[group, candidate],
                batch["adjacency"][group, candidate],
                transform,
            )[0]
    groups, candidates = nodes.shape[:2]
    return {
        **batch,
        "node_features": nodes,
        "family_ids": batch["family_ids"][None].expand(groups * candidates, -1),
    }


def _metric_set(
    logits: torch.Tensor,
    probability: torch.Tensor,
    hard: torch.Tensor,
    batch: dict,
) -> dict:
    legal = batch["legal"].cpu().numpy()
    violations = batch["violation_labels"].any(dim=-1).cpu().numpy()
    score_rows = logits.masked_fill(~hard, -torch.inf).cpu().numpy()
    ndcgs = []
    selected_violations = []
    selected_legal = []
    for index, scores in enumerate(score_rows):
        ndcgs.append(ndcg_at_k(scores, legal[index].astype(float)))
        chosen = int(np.argmax(scores))
        selected_violations.append(bool(violations[index, chosen]))
        selected_legal.append(bool(legal[index, chosen]))
    invalid_zero = bool(
        torch.equal(
            probability[~hard],
            torch.zeros_like(probability[~hard]),
        )
    )
    return {
        "ndcg_at_5": float(np.mean(ndcgs)),
        "selected_legal_rate": float(np.mean(selected_legal)),
        "selected_violation_rate": float(np.mean(selected_violations)),
        "invalid_probability_exact_zero": invalid_zero,
        "finite": bool(
            torch.isfinite(logits).all() and torch.isfinite(probability).all()
        ),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    features = _load_features(
        args.feature_root / "val/features.pt",
        device,
    )
    batch = _group_batch(features)
    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )
    model = ConstraintAwareGate(**checkpoint.get("config", {})).to(device)
    model.load_state_dict(checkpoint["model"])
    original = _score(model, batch)
    permuted_batch = _permuted(
        batch,
        torch.Generator().manual_seed(args.seed),
    )
    permuted_scores = _score(model, permuted_batch)
    results = {
        "original": _metric_set(*original, batch),
        "d4_mean": [],
        "permutation": _metric_set(
            *permuted_scores,
            permuted_batch,
        ),
        "permutation_logit_max_delta": float(
            (original[0] - permuted_scores[0]).abs().max()
        ),
        "noise": _metric_set(
            *_score(
                model,
                {
                    **batch,
                    "node_features": batch["node_features"]
                    + 0.02 * torch.randn_like(batch["node_features"]),
                },
            ),
            batch,
        ),
        "missing_control": _metric_set(
            *_score(
                model,
                {
                    **batch,
                    "node_features": batch["node_features"]
                    * (
                        torch.rand(
                            batch["node_features"].shape[:3],
                            device=device,
                        )
                        > 0.2
                    )[..., None],
                },
            ),
            batch,
        ),
    }
    for transform in range(8):
        results["d4_mean"].append(
            _metric_set(
                *_score(model, _rotated(batch, transform)),
                batch,
            )
        )
    results["d4_groups"] = results.pop("d4_mean")
    results["d4_mean"] = {
        key: float(np.mean([row[key] for row in results["d4_groups"]]))
        for key in ("ndcg_at_5", "selected_legal_rate", "selected_violation_rate")
    }
    room_report = {}
    room_count = batch["room_count"][:, 0].cpu().numpy()
    for name, lower, upper in (("1-4", 1, 4), ("5-7", 5, 7), ("8+", 8, 1000)):
        mask = torch.tensor(
            (room_count >= lower) & (room_count <= upper),
            device=device,
        )
        if not bool(mask.any()):
            continue
        subset = {
            key: value[mask] if key in GROUP_KEYS else value
            for key, value in batch.items()
        }
        room_report[name] = _metric_set(*_score(model, subset), subset)
    checks = {
        "all_finite": all(
            result["finite"]
            for key, result in results.items()
            if isinstance(result, dict) and "finite" in result
        )
        and all(row["finite"] for row in results["d4_groups"]),
        "invalid_probability_exact_zero": all(
            result["invalid_probability_exact_zero"]
            for key, result in results.items()
            if isinstance(result, dict)
            and "invalid_probability_exact_zero" in result
        )
        and all(
            row["invalid_probability_exact_zero"]
            for row in results["d4_groups"]
        ),
        "permutation_logits_stable_at_1e_5": (
            results["permutation_logit_max_delta"] <= 1.0e-5
        ),
        "d4_ndcg_drop_at_most_0_05": (
            results["original"]["ndcg_at_5"]
            - results["d4_mean"]["ndcg_at_5"]
            <= 0.05
        ),
        "stress_violation_bounded_at_0_06": all(
            results[name]["selected_violation_rate"] <= 0.06
            for name in (
                "original",
                "d4_mean",
                "permutation",
                "noise",
                "missing_control",
            )
        ),
        "no_catastrophic_operation_degradation": all(
            result["ndcg_at_5"] >= 0.50 for result in room_report.values()
        ),
    }
    report = {
        "schema_version": "residential-cagn-stress-v1",
        "status": "passed" if all(checks.values()) else "failed",
        "scope": "residential_floorplan_only_validation",
        "seed": args.seed,
        "checks": checks,
        "results": results,
        "room_count_bins": room_report,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    rows = [
        "# CAGN validation stress",
        "",
        "| Stress | NDCG@5 | Selected legal | Selected violation |",
        "|---|---:|---:|---:|",
    ]
    for name in ("original", "d4_mean", "permutation", "noise", "missing_control"):
        row = results[name]
        rows.append(
            f"| {name} | {row['ndcg_at_5']:.4f} | "
            f"{row['selected_legal_rate']:.4f} | "
            f"{row['selected_violation_rate']:.4f} |"
        )
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "checks": checks}, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
