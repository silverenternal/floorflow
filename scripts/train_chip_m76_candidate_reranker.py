#!/usr/bin/env python3
"""Train and cross-validate a target-free candidate legalizability reranker."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from floorflow.chip_flow.candidate_legalizability import (
    CandidateLegalizabilityReranker,
    legalized_cost_targets,
    listwise_ranking_loss,
    select_candidate,
    target_free_candidate_features,
)
from floorflow.chip_flow.legalizer_portfolio import select_legalizer_candidate
from floorflow.evaluation.paired_bootstrap import paired_delta
from scripts.build_chip_cross_checkpoint_portfolio import view

M48 = {
    19: "outputs/reports/chip_m76_m48_dev32_seed19_v1.json",
    20: "outputs/reports/chip_m76_m48_dev32_seed20_v1.json",
    21: "outputs/reports/chip_m76_m48_dev32_seed21_v1.json",
}
M61B = "outputs/reports/chip_m72_m61b_portfolio_dev32_v1.json"
M65 = "outputs/reports/chip_m72_m65_portfolio_dev32_v1.json"
VARIANTS = ("baseline", "expanded", "projector32")
METRICS = ("cost", "v_rel", "area_gap", "b2b_hpwl_gap")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/chip_m76_candidate_reranker_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_m76_dev_selection_v1.json",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "outputs/train/chip_m76_candidate_reranker_v1/final.pt",
    )
    return parser.parse_args()


def load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def load_path(path: Path) -> dict:
    resolved = path if path.is_absolute() else ROOT / path
    return json.loads(resolved.read_text(encoding="utf-8"))


def source_rows(path: str) -> dict[int, dict]:
    return {int(row["index"]): row for row in load(path)["rows"]}


def build_groups() -> list[dict]:
    sources = {
        f"M48-s{seed}": source_rows(path) for seed, path in M48.items()
    }
    sources["M61B"] = source_rows(M61B)
    sources["M65"] = source_rows(M65)
    indices = sorted(set.intersection(*(set(rows) for rows in sources.values())))
    if len(indices) != 32:
        raise ValueError(f"expected 32 development cases, got {len(indices)}")
    groups = []
    for index in indices:
        names = []
        entries = []
        for source, rows in sources.items():
            for variant in VARIANTS:
                names.append(f"{source}:{variant}")
                entries.append(rows[index][variant])
        groups.append(
            {
                "index": index,
                "names": names,
                "entries": entries,
                "features": target_free_candidate_features(entries),
                "costs": legalized_cost_targets(entries),
            }
        )
    return groups


def train_model(
    train_groups: list[dict],
    config: dict,
    seed: int,
) -> CandidateLegalizabilityReranker:
    torch.manual_seed(seed)
    model = CandidateLegalizabilityReranker(hidden=int(config["model"]["hidden"]))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["optimizer"]["learning_rate"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )
    temperature = float(config["loss"]["temperature"])
    epochs = int(config["validation"]["epochs_per_fold"])
    for _ in range(epochs):
        order = torch.randperm(len(train_groups))
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for group_index in order.tolist():
            group = train_groups[group_index]
            scores = model(group["features"])
            losses.append(
                listwise_ranking_loss(
                    scores,
                    group["costs"],
                    temperature=temperature,
                )
            )
        loss = torch.stack(losses).mean()
        loss.backward()
        optimizer.step()
    return model.eval()


def predictive_selection(model: CandidateLegalizabilityReranker, group: dict) -> int:
    with torch.no_grad():
        return select_candidate(model(group["features"]))


def _geometry_score(entry: dict) -> float:
    geometry = entry["target_free_geometry"]
    return (
        float(geometry["b2b_hpwl"])
        + float(geometry["p2b_hpwl"])
        + 1.0
        - float(geometry["outline_compactness"])
        + abs(float(geometry["outline_aspect"]))
    )


def safe_predictive_selection(
    model: CandidateLegalizabilityReranker,
    group: dict,
    guarded_index: int,
) -> tuple[int, bool]:
    prediction = predictive_selection(model, group)
    baseline = group["entries"][0]["legalizer"]
    guarded_score = _geometry_score(group["entries"][guarded_index])
    allowed = []
    for index, entry in enumerate(group["entries"]):
        legalizer = entry["legalizer"]
        if (
            int(legalizer["rung"]) > int(baseline["rung"])
            or int(legalizer["solves"]) > int(baseline["solves"])
        ):
            continue
        if _geometry_score(entry) <= guarded_score + 1.0e-12:
            allowed.append(index)
    if prediction in allowed:
        return prediction, False
    with torch.no_grad():
        scores = model(group["features"])
    selected = min(allowed, key=lambda index: float(scores[index]))
    return selected, selected != guarded_index


def guarded_selection(group: dict) -> int:
    baseline = view("M48-s19:baseline", group["entries"][0])
    candidates = tuple(
        view(name, entry)
        for name, entry in zip(group["names"][1:], group["entries"][1:], strict=True)
    )
    selected = select_legalizer_candidate(
        baseline,
        candidates,
        policy="legality_guarded_consensus",
    )
    return group["names"].index(selected.name)


def metric(entry: dict, key: str) -> float:
    legalized = entry["stages"]["legalized"]
    value = legalized.get(key)
    if value is None:
        value = entry["target_free_geometry"][key]
    return float(value)


def rows_for_selection(groups: list[dict], selections: list[int]) -> list[dict]:
    return [
        {
            "index": group["index"],
            "selection": group["names"][selection],
            **{key: metric(group["entries"][selection], key) for key in METRICS},
            "feasible": bool(
                group["entries"][selection]["stages"]["legalized"]["feasible"]
            ),
        }
        for group, selection in zip(groups, selections, strict=True)
    ]


def paired(current: list[dict], reference: list[dict], key: str) -> dict:
    return paired_delta(
        current,
        reference,
        key,
        iterations=10000,
        seed=7600 + METRICS.index(key),
        lower_is_better=True,
    )


def main() -> None:
    args = parse_args()
    config = load_path(args.config)
    groups = build_groups()
    folds = int(config["validation"]["protocol"].split("_")[0])
    seed = int(config["validation"]["seed"])
    predictions = [0] * len(groups)
    safety_overrides = []
    fold_rows = []
    for fold in range(folds):
        validation = [index for index in range(len(groups)) if index % folds == fold]
        training = [index for index in range(len(groups)) if index % folds != fold]
        model = train_model([groups[index] for index in training], config, seed + fold)
        for group_index in validation:
            guarded = guarded_selection(groups[group_index])
            selected, overridden = safe_predictive_selection(
                model,
                groups[group_index],
                guarded,
            )
            predictions[group_index] = selected
            if overridden:
                safety_overrides.append(groups[group_index]["index"])
        fold_rows.append(
            {
                "fold": fold,
                "train_cases": len(training),
                "validation_cases": len(validation),
            }
        )
    guarded = [guarded_selection(group) for group in groups]
    oracle = [int(torch.argmin(group["costs"]).item()) for group in groups]
    reranker_rows = rows_for_selection(groups, predictions)
    guarded_rows = rows_for_selection(groups, guarded)
    oracle_rows = rows_for_selection(groups, oracle)
    paired_results = {
        key: paired(reranker_rows, guarded_rows, key) for key in METRICS
    }
    means = {
        "reranker": {
            key: sum(row[key] for row in reranker_rows) / len(reranker_rows)
            for key in METRICS
        },
        "guarded": {
            key: sum(row[key] for row in guarded_rows) / len(guarded_rows)
            for key in METRICS
        },
        "oracle": {
            key: sum(row[key] for row in oracle_rows) / len(oracle_rows)
            for key in METRICS
        },
    }
    checks = {
        "all_reranker_feasible": all(row["feasible"] for row in reranker_rows),
        "cost_non_regression": paired_results["cost"]["relative_delta"] <= 0.0,
        "v_rel_non_regression": paired_results["v_rel"]["relative_delta"] <= 0.0,
        "area_non_regression": paired_results["area_gap"]["relative_delta"] <= 0.0,
        "b2b_non_regression": (
            paired_results["b2b_hpwl_gap"]["relative_delta"] <= 0.0
        ),
    }
    report = {
        "version": "chip-m76-dev-selection-v1",
        "status": "pass" if all(checks.values()) else "reject",
        "config": str(args.config),
        "cases": len(groups),
        "candidates_per_case": len(groups[0]["entries"]),
        "cross_validation": {
            "protocol": config["validation"]["protocol"],
            "folds": fold_rows,
        },
        "checks": checks,
        "paired_reranker_vs_guarded": paired_results,
        "means": means,
        "oracle_top1_agreement": sum(
            prediction == target for prediction, target in zip(predictions, oracle, strict=True)
        )
        / len(groups),
        "safety_override_cases": safety_overrides,
        "selected_rows": reranker_rows,
        "guarded_rows": guarded_rows,
        "oracle_rows_diagnostic_only": oracle_rows,
        "claim_boundary": (
            "Grouped case-out development screen only. The oracle row uses "
            "official cost and is diagnostic, never deployable."
        ),
    }
    final_model = train_model(groups, config, seed + folds)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": final_model.state_dict(),
            "config": config,
            "feature_names": list(config["features"]),
        },
        args.checkpoint,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "checks": checks, "means": means}, indent=2))


if __name__ == "__main__":
    main()
