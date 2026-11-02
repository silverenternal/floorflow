#!/usr/bin/env python3
"""Train a typed cross-attention aggregator over OOF atom evidence."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from floorflow.chip_flow.atom_score_heads import ATOM_NAMES
from floorflow.chip_flow.candidate_structural_selector import (
    DEFAULT_UTILITY_WEIGHTS,
    SELECTOR_METRICS,
    candidate_structural_features,
    load_structural_selector,
)
from floorflow.chip_flow.training_runtime import write_json_atomic
from floorflow.chip_flow.typed_atom_aggregator import TypedAtomAggregator

ATOM_EDGES = np.asarray(
    [
        [1, 1, 1, 0, 0, 1],
        [1, 1, 0, 1, 0, 0],
        [1, 0, 1, 0, 0, 1],
        [0, 1, 0, 1, 1, 0],
        [0, 0, 0, 1, 1, 0],
        [1, 0, 1, 0, 0, 1],
    ],
    dtype=bool,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cohort-dir",
        type=Path,
        default=Path("outputs/cohorts/chip_s205_consensus_500_v4"),
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("outputs/reports/chip_s3_candidate_legalized_labels_v3_500_v1.json"),
    )
    parser.add_argument(
        "--split-report",
        type=Path,
        default=Path("outputs/reports/chip_s205_consensus_500_split_v3.json"),
    )
    parser.add_argument(
        "--atom-oof",
        type=Path,
        default=Path("outputs/reports/chip_atom_heads_oof_v1.json"),
    )
    parser.add_argument(
        "--atom-checkpoint",
        type=Path,
        default=Path("outputs/train/chip_atom_heads_oof_v1/atom_heads.pkl"),
    )
    parser.add_argument(
        "--selector",
        type=Path,
        default=Path("outputs/train/chip_s3_structural_selector_v1/selector.pkl"),
    )
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=2.0e-3)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=251)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_labels(path: Path) -> dict[int, dict[str, dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        int(row["case_index"]): row["rows"]
        for row in payload["rows"]
    }


def _load_cases(
    cohort_dir: Path,
    labels: dict[int, dict[str, dict]],
    allowed: set[int],
) -> list[dict]:
    cases = []
    for path in sorted(cohort_dir.glob("*.npz")):
        payload = np.load(path)
        index = int(payload["case_index"])
        if index not in allowed or index not in labels:
            continue
        mask = np.asarray(payload["candidate_mask"])[0].astype(bool)
        features = np.asarray(payload["candidate_features"])[0][mask].astype(
            np.float32
        )
        names = [f"candidate_{value}" for value in range(len(features))]
        metrics = np.asarray(
            [
                [float(labels[index][name][metric]) for metric in SELECTOR_METRICS]
                for name in names
            ],
            dtype=np.float32,
        )
        cases.append(
            {
                "index": index,
                "features": features,
                "metrics": metrics,
            }
        )
    return cases


def _predict_atoms(models: dict[str, dict], features: np.ndarray) -> np.ndarray:
    expanded = candidate_structural_features(features)
    output = []
    for name in ATOM_NAMES:
        head = models[name]
        output.append(
            head["model"].predict(expanded) * head["scale"] + head["mean"]
        )
    return np.stack(output, axis=-1).astype(np.float32)


def _oof_legacy_predictions(
    cases: list[dict],
    *,
    folds: int,
    trees: int,
    seed: int,
) -> np.ndarray:
    predictions = np.zeros(
        (sum(len(case["features"]) for case in cases), len(SELECTOR_METRICS)),
        dtype=np.float32,
    )
    candidates = np.concatenate(
        [candidate_structural_features(case["features"]) for case in cases]
    )
    metrics = np.concatenate([case["metrics"] for case in cases])
    starts = np.cumsum([0] + [len(case["features"]) for case in cases])
    case_folds = KFold(
        n_splits=folds,
        shuffle=True,
        random_state=seed,
    ).split(cases)
    for fold, (train_cases, val_cases) in enumerate(case_folds):
        train_rows = np.concatenate(
            [np.arange(starts[i], starts[i + 1]) for i in train_cases]
        )
        val_rows = np.concatenate(
            [np.arange(starts[i], starts[i + 1]) for i in val_cases]
        )
        mean = metrics[train_rows].mean(axis=0)
        scale = metrics[train_rows].std(axis=0)
        scale[scale < 1.0e-8] = 1.0
        model = ExtraTreesRegressor(
            n_estimators=trees,
            max_depth=4,
            min_samples_leaf=8,
            max_features=0.7,
            random_state=seed + fold,
            n_jobs=-1,
        )
        model.fit(
            candidates[train_rows],
            (metrics[train_rows] - mean) / scale,
        )
        predictions[val_rows] = model.predict(candidates[val_rows]) * scale + mean
    return predictions


class AtomEncoder(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.atom_value = nn.Linear(1, hidden)
        self.atom_embedding = nn.Parameter(torch.zeros(len(ATOM_NAMES), hidden))
        self.context = nn.Linear(len(SELECTOR_METRICS), hidden)

    def forward(
        self,
        atom_values: torch.Tensor,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        atom_tokens = (
            self.atom_value(atom_values[..., None])
            + self.atom_embedding
        )
        context = self.context(features)[..., None, :]
        return atom_tokens, context


def _listwise(predicted: torch.Tensor, target: torch.Tensor, temperature: float) -> torch.Tensor:
    valid = torch.isfinite(target)
    safe_target = torch.where(valid, target, torch.zeros_like(target))
    probability = torch.softmax(-safe_target.detach() / temperature, dim=-1)
    log_probability = torch.log_softmax(predicted, dim=-1)
    return -(
        probability * log_probability * valid.float()
    ).sum(dim=-1).mean()


def _selective_stats(cases: list[dict], choices: list[int]) -> dict:
    joint = strict = safe = 0
    selected_rows = []
    reference_rows = []
    for case, choice in zip(cases, choices, strict=True):
        current = dict(zip(SELECTOR_METRICS, case["metrics"][choice], strict=True))
        reference = dict(zip(SELECTOR_METRICS, case["metrics"][0], strict=True))
        primary = (
            current["legalized_hard_cost"] <= reference["legalized_hard_cost"]
            and current["legalized_v_rel"] <= reference["legalized_v_rel"]
        )
        structural = (
            primary
            and current["legalized_grouping"] <= reference["legalized_grouping"]
            and current["legalized_boundary"] <= reference["legalized_boundary"]
            and current["legalized_p2b_hpwl_gap"]
            <= reference["legalized_p2b_hpwl_gap"]
        )
        constraint_safe = (
            structural
            and current["legalized_area_gap"] <= reference["legalized_area_gap"]
            and current["legalized_b2b_hpwl_gap"]
            <= reference["legalized_b2b_hpwl_gap"]
        )
        joint += int(primary)
        strict += int(structural)
        safe += int(constraint_safe)
        selected_rows.append(current)
        reference_rows.append(reference)
    relative = {
        metric: float(
            np.mean([row[metric] for row in selected_rows])
            / np.mean([row[metric] for row in reference_rows])
            - 1.0
        )
        for metric in SELECTOR_METRICS
    }
    return {
        "cases": len(cases),
        "joint_wins": joint,
        "strict_structural_wins": strict,
        "constraint_safe_wins": safe,
        "jsr": joint / len(cases),
        "csj": safe / len(cases),
        "relative": relative,
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    split = json.loads(args.split_report.read_text(encoding="utf-8"))
    labels = _load_labels(args.labels)
    train_cases = _load_cases(
        args.cohort_dir,
        labels,
        {int(value) for value in split["train_cases"]},
    )
    val_cases = _load_cases(
        args.cohort_dir,
        labels,
        {int(value) for value in split["val_cases"]},
    )
    oof = json.loads(args.atom_oof.read_text(encoding="utf-8"))
    train_indices = [int(value) for value in oof["oof_tensor"]["indices"]]
    by_index = {case["index"]: case for case in train_cases}
    train_cases = [by_index[index] for index in train_indices]
    train_atoms = np.asarray(oof["oof_tensor"]["predictions"], dtype=np.float32)
    train_atoms = np.nan_to_num(train_atoms, nan=0.0)
    atom_mean = train_atoms.mean(axis=(0, 1), keepdims=True)
    atom_scale = train_atoms.std(axis=(0, 1), keepdims=True)
    atom_scale[atom_scale < 1.0e-8] = 1.0
    train_atoms = (train_atoms - atom_mean) / atom_scale
    with args.atom_checkpoint.open("rb") as handle:
        models = pickle.load(handle)
    val_atoms = np.stack(
        [_predict_atoms(models, case["features"]) for case in val_cases]
    )
    val_atoms = (val_atoms - atom_mean[0, 0]) / atom_scale[0, 0]
    train_metric_predictions = _oof_legacy_predictions(
        train_cases,
        folds=5,
        trees=256,
        seed=args.seed + 37,
    )
    metric_mean = train_metric_predictions.mean(axis=0)
    metric_scale = train_metric_predictions.std(axis=0)
    metric_scale[metric_scale < 1.0e-8] = 1.0
    max_candidates = max(
        max(len(case["features"]) for case in train_cases),
        max(len(case["features"]) for case in val_cases),
    )
    train_atom_tensor = torch.zeros(
        len(train_cases), max_candidates, len(ATOM_NAMES)
    )
    train_context_tensor = torch.zeros(
        len(train_cases), max_candidates, len(SELECTOR_METRICS)
    )
    train_legacy = torch.zeros(len(train_cases), max_candidates)
    train_candidate_mask = torch.zeros(
        len(train_cases), max_candidates, dtype=torch.bool
    )
    cost = []
    v_rel = []
    offset = 0
    utility_weights = torch.from_numpy(DEFAULT_UTILITY_WEIGHTS.astype(np.float32))
    for row, case in enumerate(train_cases):
        count = len(case["features"])
        prediction = torch.from_numpy(
            train_metric_predictions[offset : offset + count]
        )
        offset += count
        train_atom_tensor[row, :count] = torch.from_numpy(train_atoms[row, :count])
        train_context_tensor[row, :count] = (
            prediction - torch.from_numpy(metric_mean)
        ) / torch.from_numpy(metric_scale)
        train_legacy[row, :count] = prediction @ utility_weights + 0.25 * prediction[:, 0]
        train_candidate_mask[row, :count] = True
        metric = torch.from_numpy(case["metrics"])
        cost.append(metric[:, 0])
        v_rel.append(metric[:, 1])
    cost_tensor = torch.stack(cost)
    vrel_tensor = torch.stack(v_rel)
    cost_norm = (cost_tensor - cost_tensor.mean(dim=1, keepdim=True)) / cost_tensor.std(
        dim=1,
        keepdim=True,
    ).clamp_min(1.0e-6)
    vrel_norm = (vrel_tensor - vrel_tensor.mean(dim=1, keepdim=True)) / vrel_tensor.std(
        dim=1,
        keepdim=True,
    ).clamp_min(1.0e-6)
    target = 0.5 * cost_norm + 0.5 * vrel_norm
    target = torch.where(train_candidate_mask, target, torch.nan)
    encoder = AtomEncoder(args.hidden)
    aggregator = TypedAtomAggregator(
        args.hidden,
        args.heads,
        enabled=True,
    )
    with torch.no_grad():
        aggregator.context_attention.gate.fill_(1.0)
        aggregator.atom_attention.gate.fill_(1.0)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(aggregator.parameters()),
        lr=args.learning_rate,
        weight_decay=1.0e-4,
    )
    atom_edges = torch.from_numpy(ATOM_EDGES)
    for step in range(args.steps):
        atom_tokens, context = encoder(train_atom_tensor, train_context_tensor)
        output = aggregator(
            atom_tokens,
            context,
            atom_edges=atom_edges,
            context_mask=train_candidate_mask[..., None],
            legacy_utility=train_legacy,
        )
        loss = _listwise(output.aggregate_utility, target, args.temperature)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print(f"step={step} loss={float(loss.detach()):.6f}", file=sys.stderr)
    selector = load_structural_selector(args.selector)
    val_choices = []
    with torch.no_grad():
        for case, atoms in zip(val_cases, val_atoms, strict=True):
            metric_prediction = torch.from_numpy(
                selector.predict_metrics(case["features"]).astype(np.float32)
            )[None]
            context_features = (
                metric_prediction - torch.from_numpy(metric_mean)
            ) / torch.from_numpy(metric_scale)
            legacy = (
                metric_prediction @ utility_weights
                + 0.25 * metric_prediction[..., 0]
            )
            atom_values = torch.from_numpy(atoms)[None]
            atom_tokens, context = encoder(atom_values, context_features)
            output = aggregator(
                atom_tokens,
                context,
                atom_edges=atom_edges,
                context_mask=torch.ones(
                    1,
                    len(case["features"]),
                    1,
                    dtype=torch.bool,
                ),
                legacy_utility=legacy,
            )
            val_choices.append(int(output.aggregate_utility[0].argmin()))
    selector_choices = [selector.select(case["features"]) for case in val_cases]
    report = {
        "version": "chip-typed-atom-aggregator-v1",
        "status": "complete_cpu_screen",
        "scope": "aggregator trained on 408 case-disjoint OOF atom predictions; 92-case development evaluation",
        "steps": args.steps,
        "final_loss": float(loss.detach()),
        "atom_aggregator": _selective_stats(val_cases, val_choices),
        "fixed_selector": _selective_stats(val_cases, selector_choices),
        "atom_edges": ATOM_EDGES.tolist(),
        "selection_counts": np.bincount(
            val_choices,
            minlength=max_candidates,
        ).tolist(),
    }
    write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
