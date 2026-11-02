#!/usr/bin/env python3
"""Audit target-free candidate ranking for legalized rho quality."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/Trinity/src"))

from floorflow.chip_flow.candidate_structural_selector import candidate_structural_features
from floorflow.chip_flow.rho_predictability import candidate_rank_metrics, case_flags
from floorflow.evaluation.paired_bootstrap import paired_delta

METRICS = (
    "legalized_hard_cost",
    "legalized_v_rel",
    "legalized_area_gap",
    "legalized_b2b_hpwl_gap",
    "legalized_p2b_hpwl_gap",
    "legalized_grouping",
    "legalized_boundary",
    "legalized_rho_mae",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--split-report", type=Path, required=True)
    parser.add_argument("--anchor-cases", type=Path, required=True)
    parser.add_argument("--anchor-key", default="nfe16_refine1600")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--alpha", type=float, default=10.0)
    return parser.parse_args()


def _relative(
    selected: list[dict[str, float]],
    reference: list[dict[str, float]],
) -> dict[str, float]:
    return {
        metric: float(
            np.mean([row[metric] for row in selected])
            / np.mean([row[metric] for row in reference])
            - 1.0
        )
        for metric in METRICS
    }


def _paired(
    selected: list[dict[str, float]],
    reference: list[dict[str, float]],
    iterations: int,
) -> dict[str, dict]:
    return {
        metric: paired_delta(
            selected,
            reference,
            metric,
            iterations=iterations,
            seed=31000 + offset,
            lower_is_better=True,
        )
        for offset, metric in enumerate(METRICS)
    }


def _anchor_rows(
    template: list[dict[str, float]],
    case_indices: list[int],
    path: Path,
    key: str,
) -> list[dict[str, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    anchor = {int(row["index"]): row for row in payload[key]}
    return [
        {
            metric: float(anchor[index][metric])
            for metric in METRICS
        }
        for index in case_indices
    ]


def _oracle_counts(
    candidate_rows: list[list[dict[str, float]]],
    reference_rows: list[dict[str, float]],
) -> dict[str, int]:
    four = np.asarray((0, 1, 5, 7), dtype=np.int64)
    counts = {"four_metrics": 0, "all_metrics": 0}
    for candidates, reference in zip(candidate_rows, reference_rows, strict=True):
        matrix = np.asarray(
            [[candidate[metric] for metric in METRICS] for candidate in candidates]
        )
        target = np.asarray([reference[metric] for metric in METRICS])
        counts["four_metrics"] += int(
            np.any(np.all(matrix[:, four] <= target[four], axis=1))
        )
        counts["all_metrics"] += int(np.any(np.all(matrix <= target, axis=1)))
    counts["cases"] = len(reference_rows)
    return counts


def main() -> None:
    args = parse_args()
    labels = {
        int(row["case_index"]): row
        for row in json.loads(args.labels.read_text(encoding="utf-8"))["rows"]
    }
    split = json.loads(args.split_report.read_text(encoding="utf-8"))
    train_indices = {int(value) for value in split["train_cases"]}
    validation_indices = {int(value) for value in split["val_cases"]}
    cases = {}
    for path in sorted(args.cohort_dir.glob("*.npz")):
        payload = np.load(path)
        index = int(payload["case_index"])
        mask = torch.as_tensor(payload["candidate_mask"]).bool()[0]
        features = torch.as_tensor(payload["candidate_features"]).float()[0][mask].numpy()
        cases[index] = features.astype(np.float64)

    train = [
        (index, features)
        for index, features in cases.items()
        if index in train_indices
    ]
    validation = [
        (index, features)
        for index, features in cases.items()
        if index in validation_indices
    ]
    train_x = np.concatenate(
        [candidate_structural_features(features) for _, features in train]
    )
    train_y = np.concatenate(
        [
            [
                float(rows["rows"][f"candidate_{candidate}"]["legalized_rho_mae"])
                for candidate in range(len(features))
            ]
            for index, features in train
            for rows in [labels[index]]
        ]
    )
    model = Ridge(alpha=args.alpha).fit(train_x, train_y)

    predicted = []
    actual = []
    selected_rows = []
    reference_rows = []
    candidate_rows = []
    case_indices = []
    counts = np.zeros(4, dtype=np.int64)
    for index, features in validation:
        scores = model.predict(candidate_structural_features(features))
        labels_rho = np.asarray(
            [
                labels[index]["rows"][f"candidate_{candidate}"]["legalized_rho_mae"]
                for candidate in range(len(features))
            ]
        )
        choice = int(np.argmin(scores))
        counts[choice] += 1
        predicted.append(scores)
        actual.append(labels_rho)
        case_indices.append(index)
        candidate_rows.append(
            [
                {
                    metric: float(labels[index]["rows"][f"candidate_{candidate}"][metric])
                    for metric in METRICS
                }
                for candidate in range(len(features))
            ]
        )
        selected_rows.append(candidate_rows[-1][choice])
        reference_rows.append(candidate_rows[-1][0])

    anchor_rows = _anchor_rows(
        selected_rows,
        case_indices,
        args.anchor_cases,
        args.anchor_key,
    )
    wins = {"joint": 0, "strict_structural": 0, "constraint_safe": 0}
    for candidate, reference in zip(selected_rows, anchor_rows, strict=True):
        joint, strict, safe = case_flags(candidate, reference)
        wins["joint"] += int(joint)
        wins["strict_structural"] += int(strict)
        wins["constraint_safe"] += int(safe)
    wins["cases"] = len(validation)
    report = {
        "version": "chip-rho-predictability-v1",
        "status": "diagnostic_only",
        "target_free_inference_features": True,
        "case_split": {
            "train": len(train),
            "validation": len(validation),
            "disjoint": not bool(train_indices & validation_indices),
        },
        "selection_counts": counts.tolist(),
        "rank_metrics": candidate_rank_metrics(predicted, actual),
        "selected_mean_rho": float(
            np.mean([row["legalized_rho_mae"] for row in selected_rows])
        ),
        "candidate0_mean_rho": float(
            np.mean([row["legalized_rho_mae"] for row in reference_rows])
        ),
        "rho_oracle_mean": float(np.mean([np.min(row) for row in actual])),
        "versus_candidate0": {
            "relative": _relative(selected_rows, reference_rows),
            "paired": _paired(selected_rows, reference_rows, args.bootstrap),
        },
        "versus_anchor": {
            "relative": _relative(selected_rows, anchor_rows),
            "paired": _paired(selected_rows, anchor_rows, args.bootstrap),
            "wins": wins,
        },
        "candidate_pool_oracle": _oracle_counts(candidate_rows, anchor_rows),
        "decision": (
            "rho candidate rank is predictable, but reranking cannot expand the "
            "candidate pool and does not reach anchor-relative coverage; do not deploy"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("rank_metrics", "selected_mean_rho", "candidate0_mean_rho", "rho_oracle_mean", "candidate_pool_oracle", "decision")}, indent=2))


if __name__ == "__main__":
    main()
