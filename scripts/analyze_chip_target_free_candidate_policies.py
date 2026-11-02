#!/usr/bin/env python3
"""Evaluate predeclared target-free candidate selection policies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from floorflow.evaluation.paired_bootstrap import paired_delta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort-dir", type=Path, required=True)
    parser.add_argument("--split-report", type=Path, required=True)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    result = np.empty(len(values), dtype=np.float64)
    result[order] = np.arange(len(values), dtype=np.float64)
    return result / max(len(values) - 1, 1)


def _policy_scores(states: torch.Tensor, features: torch.Tensor) -> dict[str, np.ndarray]:
    feature = features.cpu().numpy().astype(np.float64)
    state = states.cpu().numpy().astype(np.float64)
    flat = state.reshape(len(state), -1)
    distances = np.linalg.norm(flat[:, None] - flat[None], axis=-1)
    np.fill_diagonal(distances, np.inf)
    b2b = feature[:, 3].copy()
    reference = feature[0]
    safe = (
        (feature[:, 7] <= reference[7])
        & (feature[:, 8] <= reference[8])
        & (feature[:, 9] <= 1.1 * reference[9])
        & (feature[:, 10] <= 1.1 * reference[10])
    )
    guarded_b2b = b2b.copy()
    guarded_b2b[~safe] = np.inf
    guarded_b2b[0] = b2b[0]
    policies = {
        "min_b2b": feature[:, 3],
        "min_b2b_safety_guard": guarded_b2b,
        "min_p2b": feature[:, 4],
        "min_solver": 0.8 * feature[:, 7] + 0.2 * feature[:, 8],
        "min_displacement": feature[:, 10],
        "geometry_rank_sum": sum(
            _ranks(feature[:, index]) for index in (3, 4, 5, 6)
        ),
        "legalizer_rank_sum": sum(
            _ranks(feature[:, index]) for index in (7, 8, 9, 10)
        ),
        "balanced_rank_sum": sum(
            _ranks(feature[:, index]) for index in (3, 4, 5, 6, 7, 8, 9, 10)
        ),
        "latent_medoid": distances.mean(axis=1),
    }
    return policies


def main() -> None:
    args = parse_args()
    split = json.loads(args.split_report.read_text(encoding="utf-8"))
    train = {int(value) for value in split["train_cases"]}
    val = {int(value) for value in split["val_cases"]}
    cases = []
    for path in sorted(args.cohort_dir.glob("*.npz")):
        payload = np.load(path)
        index = int(payload["case_index"])
        if index not in train | val:
            continue
        mask = torch.as_tensor(payload["candidate_mask"]).bool()[0]
        states = torch.as_tensor(payload["candidate_states"]).float()[0][mask]
        features = torch.as_tensor(payload["candidate_features"]).float()[0][mask]
        quality = torch.as_tensor(payload["quality"]).float()[0][mask].numpy()
        cases.append(
            {
                "index": index,
                "split": "train" if index in train else "val",
                "quality": quality,
                "policies": _policy_scores(states, features),
            }
        )
    policy_names = list(cases[0]["policies"])
    summary = {}
    for split_name in ("train", "val"):
        selected = [case for case in cases if case["split"] == split_name]
        if not selected:
            continue
        quality_matrix = np.stack([case["quality"] for case in selected])
        summary[split_name] = {
            "cases": len(selected),
            "reference_cost": float(np.mean([case["quality"][0] for case in selected])),
            "fixed_best_view_cost": float(np.min(quality_matrix.mean(axis=0))),
            "oracle_cost": float(np.mean(np.min(quality_matrix, axis=1))),
            "policies": {},
        }
        for policy in policy_names:
            choices = [int(np.argmin(case["policies"][policy])) for case in selected]
            costs = np.asarray(
                [case["quality"][choice] for case, choice in zip(selected, choices)]
            )
            summary[split_name]["policies"][policy] = {
                "cost": float(costs.mean()),
                "relative_to_reference": float(
                    costs.mean()
                    / summary[split_name]["reference_cost"]
                    - 1.0
                ),
                "wins_vs_reference": int(
                    np.sum(costs < np.asarray([case["quality"][0] for case in selected]))
                ),
            }
    report = {
        "version": "chip-target-free-candidate-policy-screen-v1",
        "status": "complete",
        "cohort_dir": str(args.cohort_dir),
        "split_report": str(args.split_report),
        "feature_contract": {
            "indices": {
                "3": "b2b_hpwl",
                "4": "p2b_hpwl",
                "5": "outline_area",
                "6": "log_outline_aspect",
                "7": "legalizer_rung",
                "8": "legalizer_solves",
                "9": "legalizer_reshaped",
                "10": "legalization_displacement",
            },
            "target_derived_features_used": False,
        },
        "policies": policy_names,
        "summary": summary,
        "claim_boundary": (
            "Cost-only deterministic screen on a frozen cohort. "
            "Any selected policy requires one fresh joint cost/V_rel confirmation."
        ),
    }
    if args.validation is not None:
        validation = json.loads(args.validation.read_text(encoding="utf-8"))
        rows = {int(row["case_index"]): row for row in validation["rows"]}
        val_cases = [case for case in cases if case["split"] == "val"]
        metrics = (
            "legalized_hard_cost",
            "legalized_v_rel",
            "legalized_area_gap",
            "legalized_b2b_hpwl_gap",
            "legalized_p2b_hpwl_gap",
            "legalized_grouping",
            "legalized_boundary",
            "legalized_rho_mae",
            "legalizer_rung",
            "legalizer_solves",
        )
        candidate_means = {
            candidate: {
                metric: float(
                    np.mean(
                        [
                            rows[index]["rows"][f"candidate_{candidate}"][metric]
                            for index in rows
                        ]
                    )
                )
                for metric in metrics
            }
            for candidate in range(4)
        }
        best_view = min(
            candidate_means,
            key=lambda candidate: candidate_means[candidate][
                "legalized_hard_cost"
            ],
        )
        references = {
            "reference": 0,
            "fixed_best_view": best_view,
        }
        policy_validation = {}
        for policy in policy_names:
            selected_rows = []
            counts = np.zeros(4, dtype=np.int64)
            for case in val_cases:
                choice = int(np.argmin(case["policies"][policy]))
                counts[choice] += 1
                selected_rows.append(
                    {
                        **rows[case["index"]]["rows"][f"candidate_{choice}"],
                        "index": case["index"],
                    }
                )
            comparisons = {}
            for label, candidate in references.items():
                baseline = [
                    {
                        **rows[case["index"]]["rows"][f"candidate_{candidate}"],
                        "index": case["index"],
                    }
                    for case in val_cases
                ]
                relative = {
                    metric: float(
                        np.mean([row[metric] for row in selected_rows])
                        / np.mean([row[metric] for row in baseline])
                        - 1.0
                    )
                    for metric in metrics
                }
                paired = {
                    metric: paired_delta(
                        selected_rows,
                        baseline,
                        metric,
                        iterations=10000,
                        seed=20900 + 20 * policy_names.index(policy) + offset,
                        lower_is_better=True,
                    )
                    for offset, metric in enumerate(metrics)
                }
                joint = sum(
                    current["legalized_hard_cost"]
                    <= reference_row["legalized_hard_cost"]
                    and current["legalized_v_rel"]
                    <= reference_row["legalized_v_rel"]
                    for current, reference_row in zip(
                        selected_rows,
                        baseline,
                        strict=True,
                    )
                )
                strict = sum(
                    current["legalized_hard_cost"]
                    <= reference_row["legalized_hard_cost"]
                    and current["legalized_v_rel"]
                    <= reference_row["legalized_v_rel"]
                    and current["legalized_grouping"]
                    <= reference_row["legalized_grouping"]
                    and current["legalized_boundary"]
                    <= reference_row["legalized_boundary"]
                    and current["legalized_p2b_hpwl_gap"]
                    <= reference_row["legalized_p2b_hpwl_gap"]
                    for current, reference_row in zip(
                        selected_rows,
                        baseline,
                        strict=True,
                    )
                )
                safe = sum(
                    current["legalized_hard_cost"]
                    <= reference_row["legalized_hard_cost"]
                    and current["legalized_v_rel"]
                    <= reference_row["legalized_v_rel"]
                    and current["legalized_grouping"]
                    <= reference_row["legalized_grouping"]
                    and current["legalized_boundary"]
                    <= reference_row["legalized_boundary"]
                    and current["legalized_p2b_hpwl_gap"]
                    <= reference_row["legalized_p2b_hpwl_gap"]
                    and current["legalized_area_gap"]
                    <= reference_row["legalized_area_gap"]
                    and current["legalized_b2b_hpwl_gap"]
                    <= reference_row["legalized_b2b_hpwl_gap"]
                    for current, reference_row in zip(
                        selected_rows,
                        baseline,
                        strict=True,
                    )
                )
                comparisons[label] = {
                    "relative": relative,
                    "paired": paired,
                    "wins": {
                        "joint": int(joint),
                        "strict_structural": int(strict),
                        "constraint_safe": int(safe),
                        "cases": len(selected_rows),
                    },
                }
            policy_validation[policy] = {
                "selection_counts": counts.tolist(),
                "comparisons": comparisons,
            }
        report["validation"] = {
            "path": str(args.validation),
            "cases": len(val_cases),
            "best_fixed_view": f"candidate_{best_view}",
            "candidate_means": candidate_means,
            "policies": policy_validation,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
