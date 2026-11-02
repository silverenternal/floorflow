"""Audit soft-GED sensitivity against an exact graph-distance panel."""

from __future__ import annotations

import argparse
import json
from itertools import pairwise
from pathlib import Path

import torch
from scipy.stats import spearmanr

from floorflow.objectives.soft_graph_edit import soft_graph_edit_distance

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "outputs/reports/residential_soft_ged_audit_v1.json"
FEATURES = (
    ROOT
    / "outputs/control_contracts/residential_cagn_features_v1/val/features.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=FEATURES)
    parser.add_argument("--limit-groups", type=int, default=512)
    return parser.parse_args()


def _panel():
    base_tokens = torch.tensor(
        [
            [
                [0.2, 0.2, 0.2, 0.1],
                [0.2, 0.8, 0.2, 0.2],
                [0.2, 0.5, 0.8, 0.2],
                [0.1, 0.8, 0.8, 0.1],
            ]
        ]
    )
    classes = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ]
    )
    adjacency = torch.tensor(
        [
            [
                [1, 1, 0, 0],
                [1, 1, 1, 0],
                [0, 1, 1, 1],
                [0, 0, 1, 1],
            ]
        ],
        dtype=torch.float32,
    )
    variants = []
    costs = []
    for transform in range(5):
        tokens = base_tokens.clone()
        labels = classes.clone()
        edges = adjacency.clone()
        if transform >= 1:
            labels[:, 2] = torch.tensor([0.0, 1.0, 0.0])
        if transform >= 2:
            edges[:, 0, 1] = 0.0
            edges[:, 1, 0] = 0.0
        if transform >= 3:
            tokens[0, 3] = torch.tensor([0.2, 0.2, 0.2, 0.0])
        if transform >= 4:
            edges[:, 2, 3] = 0.0
            edges[:, 3, 2] = 0.0
        variants.append((tokens, labels, edges))
        costs.append(float(transform))
    return base_tokens, classes, adjacency, variants, costs


def _validation_panel(path: Path, limit: int) -> tuple[list[float], list[float]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    rows = []
    groups = {}
    for index, group in enumerate(payload["group_index"].tolist()):
        groups.setdefault(int(group), []).append(index)
    usable = [
        indices
        for indices in groups.values()
        if payload["legal"][indices].any()
        and (~payload["legal"][indices]).any()
    ][: int(limit)]
    nodes = payload["node_mask"].shape[1]
    classes = torch.eye(nodes)
    for indices in usable:
        legal_indices = [
            index for index in indices if bool(payload["legal"][index])
        ]
        target_index = legal_indices[0]
        target_active = payload["node_mask"][target_index].bool()
        target_edges = payload["adjacency"][target_index].float()
        for index in indices:
            if bool(payload["legal"][index]):
                continue
            active = payload["node_mask"][index].bool()
            edges = payload["adjacency"][index].float()
            node_cost = int((active ^ target_active).sum())
            active_union = active | target_active
            edge_mask = active_union[:, None] & active_union[None, :]
            edge_cost = float(
                torch.triu(
                    (edges - target_edges).abs() * edge_mask,
                    diagonal=1,
                ).sum()
            )
            rows.append((index, target_index, float(node_cost + edge_cost)))
    soft = []
    exact = []
    for index, target_index, exact_cost in rows:
        features = payload["node_features"][[index, target_index]].float()
        tokens = features[:, :, (1, 4, 5, 2, 7)]
        adjacency = payload["adjacency"][[index, target_index]].float()
        valid = payload["node_mask"][[index, target_index]].bool()
        value, _terms = soft_graph_edit_distance(
            tokens[:1],
            tokens[1:],
            classes[None],
            classes[None],
            adjacency[:1],
            adjacency[1:],
            predicted_valid=valid[:1],
            target_valid=valid[1:],
        )
        soft.append(float(value.detach()))
        exact.append(exact_cost)
    return soft, exact


def main() -> None:
    args = parse_args()
    base_tokens, base_classes, base_edges, variants, exact_costs = _panel()
    soft_cost = []
    term_rows = []
    for tokens, classes, edges in variants:
        value, terms = soft_graph_edit_distance(
            tokens,
            base_tokens,
            classes,
            base_classes,
            edges,
            base_edges,
        )
        soft_cost.append(float(value))
        term_rows.append({key: float(value) for key, value in terms.items()})
    rho = float(spearmanr(soft_cost, exact_costs).statistic)
    if args.features.exists():
        validation_soft, validation_exact = _validation_panel(
            args.features,
            args.limit_groups,
        )
        validation_rho = float(
            spearmanr(validation_soft, validation_exact).statistic
        )
    else:
        validation_soft = []
        validation_exact = []
        validation_rho = None
    checks = {
        "monotone_panel": all(
            right >= left
            for left, right in pairwise(soft_cost)
        ),
        "spearman_at_least_0_70": rho >= 0.70,
        "validation_available": validation_rho is not None,
        "validation_spearman_at_least_0_70": (
            validation_rho is not None and validation_rho >= 0.70
        ),
        "finite": all(torch.isfinite(torch.tensor(value)) for value in soft_cost),
        "training_proxy_only": True,
    }
    report = {
        "schema_version": "residential-soft-ged-audit-v1",
        "status": "passed" if all(checks.values()) else "failed",
        "scope": "training_proxy_only",
        "fixed_exact_cost": "unit-cost deterministic synthetic panel",
        "soft_costs": soft_cost,
        "exact_costs": exact_costs,
        "spearman_rho": rho,
        "validation_spearman_rho": validation_rho,
        "validation_rows": len(validation_soft),
        "validation_soft_costs": validation_soft,
        "validation_exact_costs": validation_exact,
        "validation_exact_definition": (
            "fixed weighted GED: unit active-room-presence edits plus "
            "upper-triangle absolute adjacency-weight edits on fixed "
            "room-class nodes"
        ),
        "terms": term_rows,
        "checks": checks,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(REPORT), "checks": checks}, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
