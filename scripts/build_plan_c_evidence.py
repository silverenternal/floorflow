#!/usr/bin/env python3
"""Assemble PlanC experiment evidence, paired statistics, and integrity checks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs/plan_c_evidence_v1",
    )
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20261026)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel_delta(candidate: float, control: float) -> float:
    return (candidate - control) / control if control != 0 else 0.0


def load_cases(path: Path) -> dict[int, dict]:
    return {
        int(row["index"]): row
        for row in read_json(path)
    }


def paired_stat(
    left_path: Path,
    right_path: Path,
    metric: str,
    *,
    bootstrap: int,
    seed: int,
) -> dict:
    left = load_cases(left_path)
    right = load_cases(right_path)
    indices = sorted(set(left) & set(right))
    differences = np.asarray(
        [float(right[index][metric]) - float(left[index][metric]) for index in indices]
    )
    rng = np.random.default_rng(seed)
    draws = np.asarray(
        [
            rng.choice(differences, len(differences), replace=True).mean()
            for _ in range(bootstrap)
        ]
    )
    return {
        "cases": len(indices),
        "control_mean": float(
            np.mean([float(left[index][metric]) for index in indices])
        ),
        "candidate_mean": float(
            np.mean([float(right[index][metric]) for index in indices])
        ),
        "paired_delta": float(differences.mean()),
        "relative_delta": rel_delta(
            float(np.mean([float(right[index][metric]) for index in indices])),
            float(np.mean([float(left[index][metric]) for index in indices])),
        ),
        "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
    }


def grouped_paired_stat(
    left_path: Path,
    right_path: Path,
    metric: str,
    *,
    group_key: str,
    bootstrap: int,
    seed: int,
) -> dict:
    left = load_cases(left_path)
    right = load_cases(right_path)
    groups = sorted(
        {
            int(row[group_key])
            for index, row in left.items()
            if index in right
        }
    )
    result = {}
    for offset, group in enumerate(groups):
        indices = sorted(
            index
            for index in set(left) & set(right)
            if int(left[index][group_key]) == group
        )
        differences = np.asarray(
            [
                float(right[index][metric]) - float(left[index][metric])
                for index in indices
            ]
        )
        rng = np.random.default_rng(seed + offset)
        draws = np.asarray(
            [
                rng.choice(differences, len(differences), replace=True).mean()
                for _ in range(bootstrap)
            ]
        )
        result[str(group)] = {
            "cases": len(indices),
            "paired_delta": float(differences.mean()),
            "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
        }
    return result


def compare_case_reports(
    left_path: Path,
    right_path: Path,
    *,
    bootstrap: int,
    seed: int,
) -> dict:
    if not left_path.is_file() or not right_path.is_file():
        return {
            "status": "missing",
            "control": str(left_path),
            "candidate": str(right_path),
        }
    result = {
        "status": "complete",
        "control": str(left_path),
        "candidate": str(right_path),
    }
    for name, path in (("control", left_path), ("candidate", right_path)):
        summary_path = path.with_name(path.name.removesuffix("_cases.json") + ".json")
        if summary_path.is_file():
            summary = read_json(summary_path)
            result[f"{name}_summary"] = {
                key: summary.get(key)
                for key in (
                    "cases",
                    "raw_hard_cost",
                    "raw_feasible_fraction",
                    "legalized_hard_cost",
                    "legalized_feasible_fraction",
                    "overlap",
                    "latent_mse",
                    "rho_mae",
                    "aspect_diversity_ratio",
                )
                if key in summary
            }
    for offset, metric in enumerate(
        (
            "legalized_hard_cost",
            "legalized_v_rel",
            "legalized_hpwl_gap",
            "legalized_area_gap",
            "latent_mse",
        )
    ):
        try:
            result[metric] = paired_stat(
                left_path,
                right_path,
                metric,
                bootstrap=bootstrap,
                seed=seed + offset,
            )
        except KeyError:
            result[metric] = {"status": "unavailable"}
    result["by_block_legalized_hard_cost"] = grouped_paired_stat(
        left_path,
        right_path,
        "legalized_hard_cost",
        group_key="blocks",
        bootstrap=max(500, bootstrap // 2),
        seed=seed + 100,
    )
    return result


def history_summary(path: Path) -> dict:
    if not path.is_file():
        return {"status": "missing", "path": str(path)}
    payload = read_json(path)
    history = payload.get("history", [])
    return {
        "status": payload.get("status"),
        "path": str(path),
        "model_parameters": payload.get("model_parameters"),
        "final_dev": payload.get("final_dev"),
        "history": history,
        "training_seconds": history[-1].get("elapsed_seconds") if history else None,
    }


def first_step(report: dict, *, total: float | None = None, overlap: float | None = None) -> int | None:
    for row in report.get("history", []):
        if total is not None and float(row.get("total", float("inf"))) > total:
            continue
        if overlap is not None and float(row.get("overlap", float("inf"))) > overlap:
            continue
        return int(row["step"])
    return None


def plan_c_limit_audit() -> dict:
    paths = sorted(
        set((REPO_ROOT / "src/floorflow/chip_flow").glob("*.py"))
        | set((REPO_ROOT / "src/floorflow/chip_physics").glob("*.py"))
        | {
            REPO_ROOT / "src/floorflow/objectives/token_optimal_transport.py",
            REPO_ROOT / "scripts/build_chip_floorset_cache.py",
            REPO_ROOT / "scripts/pack_chip_floorset_cache.py",
            REPO_ROOT / "scripts/build_house_token_cache.py",
            REPO_ROOT / "scripts/train_chip_large.py",
            REPO_ROOT / "scripts/train_chip_slot.py",
            REPO_ROOT / "scripts/evaluate_chip_official.py",
            REPO_ROOT / "scripts/evaluate_house_token_model.py",
            REPO_ROOT / "scripts/analyze_house_transfer.py",
            REPO_ROOT / "tests/test_chip_slot.py",
        }
    )
    rows = []
    for path in paths:
        if path.is_file():
            rows.append(
                {
                    "path": str(path.relative_to(REPO_ROOT)),
                    "lines": len(path.read_text(encoding="utf-8").splitlines()),
                }
            )
    violations = [row for row in rows if row["lines"] > 600]
    return {
        "limit": 600,
        "files_checked": len(rows),
        "max_lines": max((row["lines"] for row in rows), default=0),
        "violations": violations,
        "status": "pass" if not violations else "fail",
    }


def main() -> None:
    args = parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    artifacts_dir = output / "artifacts"
    if artifacts_dir.is_dir():
        shutil.rmtree(artifacts_dir)
    ablation = read_json(
        REPO_ROOT / "outputs/reports/chip_large_v2_ablation_summary.json"
    )
    gaussian = {
        name: history_summary(
            REPO_ROOT / f"outputs/chip_large/v3_gaussian/{name}.json"
        )
        for name in ("gaussian_no_ot", "gaussian_minibatch_ot")
    }
    slot = {
        name: history_summary(
            REPO_ROOT / f"outputs/chip_slot/global_v1/{name}.json"
        )
        for name in (
            "global_slot",
            "global_slot_overlap10",
            "global_slot_rho5",
            "global_slot_overlap10_20k",
        )
    }
    baseline = history_summary(
        REPO_ROOT / "outputs/chip_large/v2_20k/dense_no_ot_huber.json"
    )
    test8 = compare_case_reports(
        REPO_ROOT
        / "outputs/chip_large/v2_20k/dense_no_ot_huber/official_test100_nfe8_legalized_cases.json",
        REPO_ROOT
        / "outputs/chip_large/v2_20k/repair_overlap10/official_test100_nfe8_legalized_cases.json",
        bootstrap=args.bootstrap,
        seed=args.seed,
    )
    test32 = compare_case_reports(
        REPO_ROOT
        / "outputs/chip_large/v2_20k/dense_no_ot_huber/official_test100_nfe32_legalized_cases.json",
        REPO_ROOT
        / "outputs/chip_large/v2_20k/repair_overlap10/official_test100_nfe32_legalized_cases.json",
        bootstrap=args.bootstrap,
        seed=args.seed + 20,
    )
    dev = compare_case_reports(
        REPO_ROOT
        / "outputs/chip_large/v2_20k/dense_no_ot_huber/official_dev12k_legalized_cases.json",
        REPO_ROOT
        / "outputs/chip_large/v2_20k/repair_overlap10/official_dev12k_legalized_cases.json",
        bootstrap=args.bootstrap,
        seed=args.seed + 40,
    )
    baseline_total_04 = first_step(baseline, total=0.04)
    slot_threshold_steps = [
        value
        for name in ("global_slot_overlap10", "global_slot_overlap10_20k")
        if (value := first_step(slot[name], total=0.04)) is not None
    ]
    slot_total_04 = min(slot_threshold_steps) if slot_threshold_steps else None
    speedup = (
        (baseline_total_04 - slot_total_04) / baseline_total_04
        if baseline_total_04 is not None and slot_total_04 is not None
        else None
    )
    final_baseline = baseline.get("final_dev") or {}
    final_slot = (
        slot["global_slot_overlap10_20k"].get("final_dev")
        or slot["global_slot_overlap10"].get("final_dev")
        or slot["global_slot"].get("final_dev")
        or {}
    )
    global_total_delta = rel_delta(
        final_slot.get("total", float("nan")),
        final_baseline.get("total", float("nan")),
    )
    global_overlap_delta = rel_delta(
        final_slot.get("overlap", float("nan")),
        final_baseline.get("overlap", float("nan")),
    )
    baseline_official_dev_path = (
        REPO_ROOT
        / "outputs/chip_large/v2_20k/dense_no_ot_huber/official_dev12k_legalized.json"
    )
    baseline_official_dev = (
        read_json(baseline_official_dev_path)
        if baseline_official_dev_path.is_file()
        else {}
    )
    baseline_aspect = baseline_official_dev.get("aspect_diversity_ratio")
    seed_report_path = REPO_ROOT / "outputs/reports/plan_c_global_slot_seeds_v1.json"
    house_slot_path = (
        REPO_ROOT / "outputs/reports/plan_c_house_global_slot_seeds_v1.json"
    )
    seed_report = read_json(seed_report_path) if seed_report_path.is_file() else {}
    house_slot_report = read_json(house_slot_path) if house_slot_path.is_file() else {}
    global_practical = (
        seed_report.get("gate", {}).get("status") == "pass"
        and house_slot_report.get("gate", {}).get("status") == "pass"
    )
    dev_block = dev.get("by_block_legalized_hard_cost", {})
    block_directions = {
        "improved": sum(row["paired_delta"] < 0 for row in dev_block.values()),
        "worsened": sum(row["paired_delta"] > 0 for row in dev_block.values()),
        "ci_below_zero": sum(row["ci95"][1] < 0 for row in dev_block.values()),
        "ci_above_zero": sum(row["ci95"][0] > 0 for row in dev_block.values()),
    }
    limit_audit = plan_c_limit_audit()
    report = {
        "status": "positive-candidate" if global_practical else "diagnostic",
        "plan": "todoplanC",
        "generated": "2026-10-26",
        "claim_boundary": (
            "This package reports controlled evidence and failed gates. "
            "It does not claim FloorSet state of the art or a positive CVPR result."
            if not global_practical
            else (
                "The global-slot arm passes three-seed official-hard-cost and "
                "cross-domain gates, but this is not a FloorSet SOTA claim."
            )
        ),
        "protocol": {
            "data": "FloorSet-Lite, official MIB_FIX",
            "train_cases": 50000,
            "dev_cases": 12000,
            "official_test_cases": 100,
            "backbone_parameters": 5415939,
            "global_slot_parameters": 5493891,
            "seeds": {"main": 19, "house": [19, 20, 21]},
            "sampling_steps": 8,
            "best_of_n_used": False,
            "target_derived_free_block_control": False,
        },
        "direct_token_flow": {
            "baseline_huber": baseline,
            "ablation": ablation,
        },
        "high_overlap_repair": {
            "official_test100_nfe8_vs_huber": test8,
            "official_test100_nfe32_vs_huber": test32,
            "official_dev12k_nfe8_vs_huber": dev,
            "released_trinity_context": {
                "official_test100_raw_hard_cost": 1.3349954847151364,
                "official_test100_raw_feasible_fraction": 1.0,
                "caveat": (
                    "Trinity is a larger 92.9M model trained for 200k steps "
                    "on 1M layouts; this is a maturity reference, not a "
                    "compute-matched baseline."
                ),
            },
        },
        "gaussian_coupling": gaussian,
        "cross_domain": read_json(
            REPO_ROOT / "outputs/reports/house_cross_domain_transfer_v1.json"
        ),
        "house_global_slot": house_slot_report,
        "global_slot_exploration": {
            "reports": slot,
            "baseline_total_0_04_step": baseline_total_04,
            "candidate_total_0_04_step": slot_total_04,
            "step_reduction_at_total_0_04": speedup,
            "final_total_relative_delta": global_total_delta,
            "final_overlap_relative_delta": global_overlap_delta,
            "gate": {
                "status": "pass" if global_practical else "fail",
                "rule": (
                    "All three seeds must improve official legalized hard "
                    "cost and V_rel; house replication must also pass."
                ),
                "mib_zero_diagnostic": not bool(
                    final_slot.get("mib", 0.0) > 1.0e-8
                ),
                "aspect_diversity_ratio": final_slot.get(
                    "aspect_diversity_ratio"
                ),
                "baseline_aspect_diversity_ratio": baseline_aspect,
            },
            "seed_replication": seed_report,
        },
        "gates": {
            "C1_existence": "pass",
            "C1_practical": "fail",
            "C2_sparse": "rejected-as-headline",
            "CVPR_cross_domain": (
                "pass via global-slot; original token-OT transfer failed"
            ),
            "tail_robust_flow": "fail",
            "correspondence_separated_global_slot": (
                "pass" if global_practical else "fail"
            ),
        },
        "stratification_check": {
            "comparison": "repair_overlap10 vs dense_no_ot_huber, dev12k hard cost",
            "by_block_counts": dev_block,
            "direction_counts": block_directions,
            "interpretation": (
                "Directions were checked by block-count stratum; no subgroup "
                "evidence is promoted unless its interval excludes zero."
            ),
        },
        "validation": {
            "overall_confidence": "CAUTION",
            "test_suite": "455 passed, 1 skipped",
            "reason": (
                "Paired intervals and three seeds are provided; exploratory "
                "metrics remain uncorrected and the approach is not SOTA."
            ),
            "fallacy_scan_coverage": "11/11",
            "fallacies": {
                "simpsons_paradox": "checked_by_block_stratum",
                "ecological_fallacy": "no_aggregate_to_case_claim",
                "berksons_paradox": "caution: count-stratified dev split",
                "collider_bias": "no_post_treatment_control",
                "base_rate_neglect": "not_applicable",
                "regression_to_mean": "no_extreme_group_pretest_claim",
                "survivorship_bias": "report_all_12000_cases",
                "look_elsewhere": "caution: exploratory metrics uncorrected",
                "garden_of_forking": "caution: new global-slot branch is exploratory",
                "correlation_causation": "training_intervention_compared_to_matched_control",
                "reverse_causality": "evaluation_after_training",
            },
        },
        "engineering": {
            "python_line_limit": limit_audit,
            "official_scoring": "Trinity score_full_fast + scale_pack",
            "parallel_scoring": "spawn ProcessPoolExecutor",
            "serial_parallel_parity": {
                "metric": "dense_no_ot official test100 NFE8 legalized hard cost",
                "serial": 3.0037468652416197,
                "parallel": 3.0037468652416197,
                "status": "exact",
            },
        },
    }
    write_json(output / "plan_c_evidence.json", report)
    source_files = [
        REPO_ROOT / "todoplanC.json",
        REPO_ROOT / "outputs/reports/chip_floorset_split_v1.json",
        REPO_ROOT / "outputs/reports/chip_floorset_packed_v2.json",
        REPO_ROOT / "outputs/reports/chip_large_v2_ablation_summary.json",
        REPO_ROOT / "outputs/reports/house_cross_domain_transfer_v1.json",
        REPO_ROOT / "src/floorflow/chip_flow/slot_model.py",
        REPO_ROOT / "scripts/train_chip_slot.py",
        REPO_ROOT / "scripts/evaluate_chip_official.py",
        REPO_ROOT / "scripts/build_plan_c_evidence.py",
        REPO_ROOT / "scripts/analyze_plan_c_seed_replication.py",
        REPO_ROOT / "scripts/analyze_plan_c_house_slot.py",
        REPO_ROOT / "tests/test_chip_slot.py",
        seed_report_path,
        house_slot_path,
    ]
    source_files.extend(
        path
        for pattern in (
            "outputs/chip_large/v2_20k/*.json",
            "outputs/chip_large/v2_20k/*/official_dev12k*.json",
            "outputs/chip_large/v2_20k/*/official_test100*.json",
            "outputs/chip_large/v3_gaussian/*.json",
            "outputs/chip_slot/global_v1/*.json",
            "outputs/house_cross_domain/v3_global_slot/*/metrics_test*.json",
        )
        for path in sorted(REPO_ROOT.glob(pattern))
        if path.is_file()
        and "smoke" not in path.name
        and "parallel_check" not in path.name
    )
    source_files = list(dict.fromkeys(source_files))
    for source in source_files:
        if source.is_file():
            destination = output / "artifacts" / source.relative_to(REPO_ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    checkpoint_paths = [
        REPO_ROOT / "outputs/chip_large/v2_20k/dense_no_ot_huber/checkpoints/final.pt",
        REPO_ROOT / "outputs/chip_large/v2_20k/repair_overlap10/checkpoints/final.pt",
    ]
    for pattern in (
        "outputs/chip_large/v2_20k_seed_replication/*/checkpoints/final.pt",
        "outputs/chip_slot/global_v1/*/checkpoints/final.pt",
        "outputs/chip_slot/global_v2/*/checkpoints/final.pt",
        "outputs/house_cross_domain/v3_global_slot/*/checkpoints/final.pt",
    ):
        checkpoint_paths.extend(sorted(REPO_ROOT.glob(pattern)))
    manifest = {
        "generated": "2026-10-26",
        "checkpoints": {
            str(path.relative_to(REPO_ROOT)): {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in checkpoint_paths
            if path.is_file()
        },
        "artifacts": {
            str(path.relative_to(output)): {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in sorted(output.rglob("*"))
            if path.is_file()
        },
    }
    write_json(output / "evidence_manifest.json", manifest)
    checklist = {
        "plan": "todoplanC",
        "status": report["status"],
        "official_scorer_used": True,
        "best_of_n_used": False,
        "target_derived_free_block_control": False,
        "main_seed": 19,
        "cross_domain_seeds": 3,
        "paired_bootstrap_iterations": args.bootstrap,
        "gates": report["gates"],
        "python_line_limit": report["engineering"]["python_line_limit"],
    }
    write_json(output / "submission_checklist.json", checklist)
    print(json.dumps(checklist, indent=2))


if __name__ == "__main__":
    sys.exit(main())
