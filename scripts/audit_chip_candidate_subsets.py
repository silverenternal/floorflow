#!/usr/bin/env python3
"""Audit budget-matched candidate subsets on validation only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from floorflow.chip_flow.candidate_structural_selector import save_structural_selector
from floorflow.chip_flow.candidate_subset_audit import (
    budget_matched_subsets,
    validation_rank,
)
from floorflow.chip_flow.training_runtime import write_json_atomic
from scripts.train_chip_candidate_structural_selector import (
    _anchor_comparison,
    _cohort_cases,
    _evaluate_selection,
    _fit,
    _label_rows,
    _select,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--split-report", type=Path, required=True)
    parser.add_argument("--anchor-cases", type=Path, required=True)
    parser.add_argument("--anchor-key", default="nfe16_refine1600")
    parser.add_argument("--candidate-names", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trees", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--min-samples-leaf", type=int, default=8)
    parser.add_argument("--max-features", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    return parser.parse_args()


def _name_for_subset(
    subset: tuple[int, ...],
    names: list[str],
) -> str:
    return "subset_" + "_".join(names[index] for index in subset)


def main() -> None:
    args = parse_args()
    names = [value.strip() for value in args.candidate_names.split(",")]
    if not all(names):
        raise ValueError("candidate names must not be empty")
    split = json.loads(args.split_report.read_text(encoding="utf-8"))
    train_indices = {int(value) for value in split["train_cases"]}
    validation_indices = {int(value) for value in split["val_cases"]}
    overlap = train_indices & validation_indices
    if overlap:
        raise ValueError(f"train/validation overlap: {sorted(overlap)[:8]}")
    labels = _label_rows(args.labels)
    fit_args = SimpleNamespace(
        trees=args.trees,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        seed=args.seed,
        bootstrap_iterations=args.bootstrap_iterations,
    )
    results = []
    for subset in budget_matched_subsets(len(names), budget=4):
        name = _name_for_subset(subset, names)
        train = _cohort_cases(
            args.cohort_dir,
            labels,
            train_indices,
            candidate_indices=subset,
        )
        validation = _cohort_cases(
            args.cohort_dir,
            labels,
            validation_indices,
            candidate_indices=subset,
        )
        selector = _fit(train, fit_args)
        checkpoint = args.output_dir / name / "selector.pkl"
        save_structural_selector(checkpoint, selector)
        choices, selected_rows = _select(selector, validation)
        validation_report = _evaluate_selection(
            validation,
            choices,
            fit_args,
            0,
        )
        validation_report["versus_anchor"] = _anchor_comparison(
            selected_rows,
            validation,
            args.anchor_cases,
            args.anchor_key,
            fit_args,
        )
        result = {
            "name": name,
            "candidate_indices": list(subset),
            "candidate_names": [names[index] for index in subset],
            "checkpoint": str(checkpoint),
            "validation": validation_report,
        }
        result["validation_rank"] = list(validation_rank(result))
        results.append(result)
    winner = max(results, key=validation_rank)
    report = {
        "version": "chip-candidate-subset-validation-audit-v1",
        "status": "validation_only",
        "selection_contract": (
            "train 408 / validation 92; candidate 0 retained; four views; "
            "fixed selector hyperparameters; fresh500 not read"
        ),
        "candidate_names": names,
        "anchor": {
            "path": str(args.anchor_cases),
            "key": args.anchor_key,
        },
        "winner": winner["name"],
        "results": results,
    }
    write_json_atomic(args.output, report)
    print(
        json.dumps(
            {
                "winner": winner["name"],
                "winner_candidates": winner["candidate_names"],
                "winner_anchor_wins": winner["validation"]["versus_anchor"][
                    "wins"
                ],
                "winner_oracle": winner["validation"]["versus_anchor"][
                    "candidate_pool_oracle"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
