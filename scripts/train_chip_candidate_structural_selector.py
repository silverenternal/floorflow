#!/usr/bin/env python3
"""Train and confirm a target-free legalized structural candidate selector."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from floorflow.chip_flow.candidate_manifest import TARGET_FREE_FEATURE_NAMES
from floorflow.chip_flow.candidate_structural_selector import (
    SELECTOR_METRICS,
    StructuralSelectorCheckpoint,
    candidate_structural_features,
    save_structural_selector,
)
from floorflow.chip_flow.training_runtime import write_json_atomic
from floorflow.evaluation.paired_bootstrap import paired_delta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--split-report", type=Path, required=True)
    parser.add_argument("--fresh-cohort-dir", type=Path)
    parser.add_argument("--fresh-labels", type=Path)
    parser.add_argument("--fresh-anchor-cases", type=Path)
    parser.add_argument("--anchor-key", default="nfe4_refine1600")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trees", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--min-samples-leaf", type=int, default=8)
    parser.add_argument("--max-features", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    return parser.parse_args()


def _label_rows(path: Path) -> dict[int, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(row["case_index"]): row for row in payload["rows"]}


def _cohort_cases(
    cohort_dir: Path,
    labels: dict[int, dict],
    allowed: set[int] | None = None,
    candidate_indices: Sequence[int] | None = None,
) -> list[dict]:
    cases = []
    for path in sorted(cohort_dir.glob("*.npz")):
        payload = np.load(path)
        index = int(payload["case_index"])
        if index not in labels or (allowed is not None and index not in allowed):
            continue
        mask = np.asarray(payload["candidate_mask"])[0].astype(bool)
        all_features = np.asarray(payload["candidate_features"])[0][mask].astype(
            np.float64
        )
        indices = (
            list(range(len(all_features)))
            if candidate_indices is None
            else [int(value) for value in candidate_indices]
        )
        if not indices or len(indices) != len(set(indices)):
            raise ValueError("candidate indices must be non-empty and unique")
        if min(indices) < 0 or max(indices) >= len(all_features):
            raise ValueError(f"case {index} candidate index is out of range")
        candidates = [f"candidate_{candidate}" for candidate in indices]
        if any(name not in labels[index]["rows"] for name in candidates):
            raise ValueError(f"case {index} candidate labels are incomplete")
        features = all_features[indices]
        metrics = np.asarray(
            [
                [float(labels[index]["rows"][name][metric]) for metric in SELECTOR_METRICS]
                for name in candidates
            ],
            dtype=np.float64,
        )
        cases.append(
            {
                "index": index,
                "features": features,
                "metrics": metrics,
                "candidate_names": candidates,
            }
        )
    if not cases:
        raise ValueError("cohort selection produced no cases")
    return cases


def _fit(
    cases: list[dict],
    args: argparse.Namespace,
) -> StructuralSelectorCheckpoint:
    features = np.concatenate(
        [candidate_structural_features(case["features"]) for case in cases],
        axis=0,
    )
    metrics = np.concatenate([case["metrics"] for case in cases], axis=0)
    metric_mean = metrics.mean(axis=0)
    metric_scale = metrics.std(axis=0)
    metric_scale[metric_scale < 1.0e-8] = 1.0
    regressor = ExtraTreesRegressor(
        n_estimators=args.trees,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        random_state=args.seed,
        n_jobs=1,
    )
    regressor.fit(features, (metrics - metric_mean) / metric_scale)
    return StructuralSelectorCheckpoint(
        regressor=regressor,
        metric_mean=metric_mean,
        metric_scale=metric_scale,
        metadata={
            "trees": args.trees,
            "max_depth": args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
            "max_features": args.max_features,
            "seed": args.seed,
            "feature_contract": list(TARGET_FREE_FEATURE_NAMES),
            "target_free_inference": True,
        },
    )


def _select(
    selector: StructuralSelectorCheckpoint,
    cases: list[dict],
) -> tuple[list[int], list[dict]]:
    choices = [selector.select(case["features"]) for case in cases]
    rows = [
        {
            "index": case["index"],
            "case_index": case["index"],
            "candidate": choice,
            "candidate_name": case["candidate_names"][choice],
            **dict(zip(SELECTOR_METRICS, case["metrics"][choice], strict=True)),
        }
        for case, choice in zip(cases, choices, strict=True)
    ]
    return choices, rows


def _rows_for_case_choice(cases: list[dict], choices: list[int]) -> list[dict]:
    return [
        {
            "index": case["index"],
            "case_index": case["index"],
            **dict(zip(SELECTOR_METRICS, case["metrics"][choice], strict=True)),
        }
        for case, choice in zip(cases, choices, strict=True)
    ]


def _paired(
    selected: list[dict],
    baseline: list[dict],
    args: argparse.Namespace,
    seed_offset: int,
) -> dict[str, dict]:
    return {
        metric: paired_delta(
            selected,
            baseline,
            metric,
            iterations=args.bootstrap_iterations,
            seed=2300 + seed_offset + offset,
            lower_is_better=True,
        )
        for offset, metric in enumerate(SELECTOR_METRICS)
    }


def _relative(selected: list[dict], baseline: list[dict]) -> dict[str, float]:
    return {
        metric: float(
            np.mean([row[metric] for row in selected])
            / np.mean([row[metric] for row in baseline])
            - 1.0
        )
        for metric in SELECTOR_METRICS
    }


def _wins(selected: list[dict], baseline: list[dict]) -> dict[str, int | float | str]:
    joint = strict = safe = 0
    for current, reference in zip(selected, baseline, strict=True):
        primary = (
            current["legalized_hard_cost"] <= reference["legalized_hard_cost"]
            and current["legalized_v_rel"] <= reference["legalized_v_rel"]
        )
        structural = (
            primary
            and current["legalized_grouping"] <= reference["legalized_grouping"]
            and current["legalized_boundary"] <= reference["legalized_boundary"]
            and current["legalized_p2b_hpwl_gap"] <= reference["legalized_p2b_hpwl_gap"]
        )
        constraint_safe = (
            structural
            and current["legalized_area_gap"] <= reference["legalized_area_gap"]
            and current["legalized_b2b_hpwl_gap"] <= reference["legalized_b2b_hpwl_gap"]
        )
        joint += int(primary)
        strict += int(structural)
        safe += int(constraint_safe)
    return {
        "joint": joint,
        "strict_structural": strict,
        "constraint_safe": safe,
        "cases": len(selected),
        "joint_rate": joint / len(selected),
        "strict_structural_rate": strict / len(selected),
        "constraint_safe_rate": safe / len(selected),
    }


def _compare(
    selected: list[dict],
    baseline: list[dict],
    args: argparse.Namespace,
    seed_offset: int,
) -> dict:
    return {
        "relative": _relative(selected, baseline),
        "paired": _paired(selected, baseline, args, seed_offset),
        "wins": _wins(selected, baseline),
    }


def _evaluate_selection(
    cases: list[dict],
    choices: list[int],
    args: argparse.Namespace,
    seed_offset: int,
) -> dict:
    selected = _rows_for_case_choice(cases, choices)
    reference_choices = [0] * len(cases)
    b2b_choices = [int(np.argmin(case["features"][:, 3])) for case in cases]
    reference = _rows_for_case_choice(cases, reference_choices)
    min_b2b = _rows_for_case_choice(cases, b2b_choices)
    return {
        "cases": len(cases),
        "selection_counts": np.bincount(choices, minlength=4).tolist(),
        "means": {
            metric: float(np.mean([row[metric] for row in selected]))
            for metric in SELECTOR_METRICS
        },
        "versus_reference": _compare(selected, reference, args, seed_offset),
        "versus_min_b2b": _compare(selected, min_b2b, args, seed_offset + 20),
    }


def _anchor_comparison(
    selected: list[dict],
    cases: list[dict],
    path: Path,
    key: str,
    args: argparse.Namespace,
) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    anchor = {int(row["index"]): row for row in payload[key]}
    missing = sorted({row["index"] for row in selected} - set(anchor))
    if missing:
        raise ValueError(f"anchor cases are missing indices: {missing[:10]}")
    baseline = [
        {
            "index": row["index"],
            "case_index": row["index"],
            **{metric: float(anchor[row["index"]][metric]) for metric in SELECTOR_METRICS},
        }
        for row in selected
    ]
    primary_indices = np.asarray((0, 1, 5, 7), dtype=np.int64)
    primary_cases = 0
    all_metric_cases = 0
    case_by_index = {case["index"]: case for case in cases}
    for row in selected:
        metrics = case_by_index[row["index"]]["metrics"]
        anchor_metrics = np.asarray([anchor[row["index"]][metric] for metric in SELECTOR_METRICS])
        primary_cases += int(np.any(np.all(metrics[:, primary_indices] <= anchor_metrics[primary_indices], axis=1)))
        all_metric_cases += int(np.any(np.all(metrics <= anchor_metrics[None], axis=1)))
    return {
        "path": str(path),
        "key": key,
        "candidate_pool_oracle": {
            "cost_v_rel_grouping_rho": primary_cases,
            "all_metrics": all_metric_cases,
            "cases": len(selected),
        },
        **(_compare(selected, baseline, args, 60)),
    }


def main() -> None:
    args = parse_args()
    split = json.loads(args.split_report.read_text(encoding="utf-8"))
    train_indices = {int(value) for value in split["train_cases"]}
    validation_indices = {int(value) for value in split["val_cases"]}
    if train_indices & validation_indices:
        raise ValueError("training and validation cases overlap")
    labels = _label_rows(args.labels)
    train = _cohort_cases(args.cohort_dir, labels, train_indices)
    validation = _cohort_cases(args.cohort_dir, labels, validation_indices)
    selector = _fit(train, args)
    save_structural_selector(args.checkpoint, selector)

    validation_choices, _ = _select(selector, validation)
    validation_report = _evaluate_selection(
        validation,
        validation_choices,
        args,
        0,
    )

    fresh_report = None
    if args.fresh_cohort_dir is not None or args.fresh_labels is not None:
        if args.fresh_cohort_dir is None or args.fresh_labels is None:
            raise ValueError("fresh cohort and labels must be provided together")
        fresh_labels = _label_rows(args.fresh_labels)
        fresh = _cohort_cases(args.fresh_cohort_dir, fresh_labels)
        overlap = (
            {case["index"] for case in fresh}
            & (train_indices | validation_indices)
        )
        if overlap:
            raise ValueError(
                f"fresh cohort overlaps development cases: {sorted(overlap)[:10]}"
            )
        fresh_choices, fresh_rows = _select(selector, fresh)
        fresh_report = _evaluate_selection(fresh, fresh_choices, args, 40)
        if args.fresh_anchor_cases is not None:
            fresh_report["versus_anchor"] = _anchor_comparison(
                fresh_rows,
                fresh,
                args.fresh_anchor_cases,
                args.anchor_key,
                args,
            )

    candidate_budget = len(train[0]["metrics"])
    report = {
        "version": "chip-structural-candidate-selector-v1",
        "status": (
            "complete_confirmation" if fresh_report is not None else "development_only"
        ),
        "cohort_dir": str(args.cohort_dir),
        "fresh_cohort_dir": (
            str(args.fresh_cohort_dir) if args.fresh_cohort_dir is not None else None
        ),
        "split_report": str(args.split_report),
        "checkpoint": str(args.checkpoint),
        "model": {
            "family": "ExtraTreesRegressor",
            "multioutput_metrics": list(SELECTOR_METRICS),
            "target_free_features": list(TARGET_FREE_FEATURE_NAMES),
            "candidate_selection_budget": candidate_budget,
            "target_derived_inference_features": False,
            "metadata": selector.metadata,
            "utility_weights": selector.utility_weights.tolist(),
            "cost_tiebreak": selector.cost_tiebreak,
        },
        "train": {
            "cases": len(train),
            "candidate_rows": sum(len(case["metrics"]) for case in train),
        },
        "validation": validation_report,
        "fresh_holdout": fresh_report,
        "claim_boundary": (
            "Feature selection was fitted on the 408-case train split and frozen before "
            "any reported confirmation. Candidate generation and all legalizer calls "
            "must be charged to the portfolio budget."
        ),
    }
    write_json_atomic(args.output, report)
    print(
        json.dumps(
            {
                "validation": validation_report,
                "fresh_holdout": fresh_report,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
