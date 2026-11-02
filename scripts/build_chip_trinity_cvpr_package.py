#!/usr/bin/env python3
"""Assemble the three-seed graph-token flow CVPR evidence package."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_chip_cvpr_package import digest, python_file_sizes, read, write

EVIDENCE = (
    "outputs/reports/chip_trinity_token_three_seed_v1.json",
    "outputs/reports/chip_trinity_claim_gate_v1.json",
    "outputs/reports/chip_trinity_parity_gap_v1.json",
    "outputs/reports/chip_trinity_token_bridge_cpu_v1.json",
    "outputs/reports/chip_condition_slot_identity_probe_v1.json",
    "outputs/reports/chip_condition_slot_identity_probe_v2.json",
    "outputs/reports/chip_trinity_token_flow_screen_v1.json",
    "outputs/reports/chip_trinity_token_flow_20k_rescue_v1.json",
    "outputs/reports/trinity_d192_matched_protocol_v2.json",
    "outputs/reports/chip_graph_mechanism_ablation_v1.json",
    "outputs/reports/chip_global_pool_stats_screen_v1.json",
    "outputs/reports/chip_global_film_screen_v1.json",
    "outputs/reports/chip_path_consistency_screen_v1.json",
    "outputs/reports/chip_refiner_target_probe_v1.json",
    "outputs/reports/chip_inference_ensemble_screen_v1.json",
    "outputs/reports/chip_range_warm_restart_v1.json",
    "outputs/reports/chip_slot_predictability_probe_v1.json",
    "outputs/reports/chip_token_flow_component_rows_v2.json",
    "outputs/reports/chip_token_flow_mechanism_diagnostics_v1.json",
    "outputs/reports/chip_token_flow_refiner_curve_v1.json",
    "outputs/reports/chip_token_flow_budget_curve_v1.json",
    "outputs/reports/chip_token_flow_claim_gate_v2.json",
    "outputs/reports/chip_token_flow_claim_gate_v3.json",
    "outputs/reports/chip_metric_triage_v1.json",
    "outputs/reports/chip_tail_budget_mechanism_v1.json",
    "outputs/reports/chip_trinity_gap_decomposition_v1.json",
    "outputs/reports/trinity_d192_component_seed19_v1.json",
    "outputs/reports/trinity_d192_component_seed20_v1.json",
    "outputs/reports/trinity_d192_component_seed21_v1.json",
    "outputs/reports/chip_trinity_d192_component_gap_v1.json",
    "outputs/reports/chip_token_flow_pareto_v1.json",
)
SOURCE_FILES = (
    "chipstodo.json",
    "docs/chip_cvpr_status_20261027.md",
    "docs/chip_floorplanning_concepts.md",
    "docs/chip_claim_boundaries_v2.md",
    "src/floorflow/chip_flow/trinity_token_flow.py",
    "src/floorflow/chip_flow/trinity_release.py",
    "src/floorflow/chip_flow/flow_adapter.py",
    "src/floorflow/chip_flow/train_cli.py",
    "src/floorflow/chip_flow/train_objective.py",
    "src/floorflow/chip_flow/train_evaluation.py",
    "src/floorflow/chip_flow/training_runtime.py",
    "src/floorflow/chip_flow/graph_conditioning.py",
    "src/floorflow/chip_flow/projections.py",
    "src/floorflow/chip_flow/score_diagnostics.py",
    "scripts/train_chip_slot.py",
    "scripts/evaluate_chip_official.py",
    "scripts/evaluate_chip_refiner.py",
    "scripts/audit_chip_condition_slot_identity.py",
    "scripts/audit_chip_trinity_parity.py",
    "scripts/audit_chip_trinity_token_bridge.py",
    "scripts/build_chip_trinity_token_screen_report.py",
    "scripts/build_chip_trinity_token_three_seed_report.py",
    "scripts/build_chip_trinity_claim_gate.py",
    "scripts/collect_chip_token_flow_diagnostics.py",
    "scripts/merge_chip_token_flow_diagnostics.py",
    "scripts/analyze_chip_token_flow_mechanism.py",
    "scripts/collect_chip_token_flow_refiner_curve.py",
    "scripts/analyze_chip_token_flow_budget.py",
    "scripts/build_chip_token_flow_claim_gate_v2.py",
    "scripts/eval_trinity_baseline.py",
    "scripts/analyze_chip_trinity_component_gap.py",
    "scripts/build_chip_token_flow_pareto.py",
    "scripts/build_chip_token_flow_claim_gate_v3.py",
    "scripts/build_chip_tail_budget_mechanism.py",
    "scripts/build_chip_trinity_gap_decomposition.py",
    "scripts/build_chip_trinity_cvpr_package.py",
    "src/floorflow/evaluation/chip_gap.py",
    "tests/test_chip_gap.py",
    "tests/test_chip_score_diagnostics.py",
    "tests/test_chip_trinity_release_loader.py",
    "tests/test_chip_trinity_token_flow.py",
    "tests/test_chip_graph_conditioning.py",
    "tests/test_chip_sparse_hierarchy.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/cvpr_package_v13_chips",
    )
    return parser.parse_args()


def git_commit(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def pytest_summary(path: Path) -> dict[str, int]:
    root = ET.parse(path).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    if suite is None:
        raise ValueError(f"{path} has no testsuite element")
    return {
        key: int(suite.attrib.get(key, 0))
        for key in ("tests", "failures", "errors", "skipped")
    }


def max_training_seconds(root: Path) -> dict[str, float]:
    output = {}
    for seed in (19, 20, 21):
        report = read(root / f"trinity_token_seed{seed}_20k.json")
        output[str(seed)] = float(report["history"][-1]["elapsed_seconds"])
    return output


def main() -> None:
    args = parse_args()
    paths = [ROOT / relative for relative in EVIDENCE]
    missing = [str(path.relative_to(ROOT)) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing evidence: {missing}")
    primary = read(ROOT / EVIDENCE[0])
    claim = read(ROOT / "outputs/reports/chip_token_flow_claim_gate_v3.json")
    parity = read(ROOT / EVIDENCE[2])
    bridge = read(ROOT / EVIDENCE[3])
    external = read(ROOT / EVIDENCE[8])
    mechanism = read(ROOT / EVIDENCE[9])
    flow_mechanism = read(
        ROOT / "outputs/reports/chip_token_flow_mechanism_diagnostics_v1.json"
    )
    budget = read(
        ROOT / "outputs/reports/chip_token_flow_budget_curve_v1.json"
    )
    trinity_gap = read(
        ROOT / "outputs/reports/chip_trinity_d192_component_gap_v1.json"
    )
    pareto = read(
        ROOT / "outputs/reports/chip_token_flow_pareto_v1.json"
    )
    triage = read(
        ROOT / "outputs/reports/chip_metric_triage_v1.json"
    )
    tail_budget = read(
        ROOT / "outputs/reports/chip_tail_budget_mechanism_v1.json"
    )
    gap_decomposition = read(
        ROOT / "outputs/reports/chip_trinity_gap_decomposition_v1.json"
    )
    pytest_path = ROOT / "outputs/reports/chip_pytest_v4.xml"
    tests = pytest_summary(pytest_path)
    sizes = python_file_sizes()
    oversized = {path: lines for path, lines in sizes.items() if lines >= 600}
    training_seconds = max_training_seconds(
        ROOT / "outputs/chip_trinity/rescue_20k"
    )
    claim_metrics = claim["main_tables"]
    efficiency = {
        "candidate_parameters": parity["trinity"]["parameters"],
        "graph_baseline_parameters": parity["current"]["parameters"],
        "parameter_relative_gap": parity["gap"]["parameter_relative"],
        "training_seconds": training_seconds,
        "official_100_sample_seconds": {
            "raw_nfe8_by_seed": {
                str(seed): read(
                    ROOT
                    / "outputs/chip_trinity/rescue_20k"
                    / f"trinity_token_seed{seed}_20k"
                    / "official_test100_nfe8_legalized.json"
                )["evaluation_seconds"]
                for seed in (19, 20, 21)
            },
            "nfe4_refine400_sample_stage_by_seed": {
                str(seed): read(
                    ROOT
                    / "outputs/chip_trinity/rescue_20k"
                    / f"trinity_token_seed{seed}_20k"
                    / "refiner_nfe4_ref400.json"
                )["results"]["nfe4_refine400"]["stage_seconds"]["sample"]
                for seed in (19, 20, 21)
            },
        },
        "speed_claim": (
            "No cross-machine speed claim. Times are retained for reproducibility."
        ),
    }
    negative = {
        "learned_condition_slots_v1": read(ROOT / EVIDENCE[4]),
        "learned_condition_slots_v2": read(ROOT / EVIDENCE[5]),
        "token_flow_5k_screen": read(ROOT / EVIDENCE[6]),
        "global_pool_stats": read(ROOT / EVIDENCE[10]),
        "global_film": read(ROOT / EVIDENCE[11]),
        "path_consistency": read(ROOT / EVIDENCE[12]),
        "refiner_target": read(ROOT / EVIDENCE[13]),
        "inference_ensemble": read(ROOT / EVIDENCE[14]),
        "range_warm_restart": read(ROOT / EVIDENCE[15]),
        "slot_predictability": read(ROOT / EVIDENCE[16]),
    }
    checks = {
        "claim_gate_selected": claim["status"] == "selected",
        "three_seed_both_metrics_all_seeds": primary["checks"][
            "candidate_wins_both_refined_metrics_all_seeds"
        ],
        "seed_mean_refined_both_metrics": primary["checks"][
            "seed_mean_refined_improves_both"
        ],
        "cpu_bridge_passed": bridge["status"] == "pass",
        "parameter_match_within_1pct": (
            parity["gap"]["parameter_relative"] < 0.01
        ),
        "mechanism_ablation_retained": mechanism["status"] == "pass",
        "external_comparator_reported": external["mean_cost"] > 0.0,
        "flow_mechanism_supported": all(
            flow_mechanism["checks"][key]
            for key in (
                "structural_component_supported",
                "tail_improvement_supported",
            )
        ),
        "equal_quality_budget_supported": all(budget["checks"].values()),
        "trinity_gap_componentized": trinity_gap["status"]
        == "pass_with_legacy_reproducibility_warning",
        "gap_decomposition_passed": gap_decomposition["status"] == "pass",
        "gap_decomposition_checks": all(
            gap_decomposition["checks"].values()
        ),
        "metric_triage_passed": all(triage["checks"].values()),
        "tail_budget_mechanism_passed": (
            tail_budget["status"] == "pass"
            and all(tail_budget["checks"].values())
        ),
        "legacy_external_anchor_retained": claim["external_anchor"][
            "legacy_no_seed"
        ]["candidate_relative_gap"]
        == trinity_gap["legacy_anchor"]["candidate_relative_gap"],
        "pareto_assets_present": all(
            (ROOT / relative).is_file()
            for relative in (
                "outputs/figures/chip_token_flow_pareto_v1.svg",
                "outputs/figures/chip_token_flow_pareto_v1.pdf",
            )
        ),
        "negative_results_retained": (
            len(negative) == 10
            and all(isinstance(value, dict) for value in negative.values())
        ),
        "engineering_tests_pass": (
            tests["failures"] == 0 and tests["errors"] == 0
        ),
        "source_and_tests_available": all(
            (ROOT / relative).is_file() for relative in SOURCE_FILES
        ),
        "row_level_evidence_available": all(path.is_file() for path in paths),
        "no_python_file_at_or_over_600": not oversized,
    }
    manifest = {
        "root": str(ROOT),
        "evidence": {
            str(path.relative_to(ROOT)): digest(path) for path in paths
        },
        "sources": {
            relative: digest(ROOT / relative)
            for relative in dict.fromkeys(SOURCE_FILES)
        },
        "figures": {
            relative: digest(ROOT / relative)
            for relative in (
                "outputs/figures/chip_token_flow_pareto_v1.svg",
                "outputs/figures/chip_token_flow_pareto_v1.pdf",
                "paper/figures/chip_tail_budget_mechanism.svg",
                "paper/figures/chip_tail_budget_mechanism.pdf",
                "paper/figures/chip_trinity_gap_decomposition.svg",
                "paper/figures/chip_trinity_gap_decomposition.pdf",
            )
        },
        "third_party_trinity": {
            "commit": git_commit(ROOT / "third_party/Trinity"),
            "config_sha256": digest(
                ROOT / "third_party/Trinity/checkpoints/d192/config.json"
            ),
        },
        "pytest": {
            "path": str(pytest_path.relative_to(ROOT)),
            "sha256": digest(pytest_path),
            "summary": tests,
        },
        "python_file_max_lines": max(sizes.values()),
        "python_files_600_or_more": oversized,
    }
    checklist = {
        "status": "ready" if all(checks.values()) else "blocked",
        "scope": "CVPR experiment evidence package; not manuscript text",
        "checks": checks,
        "claim": claim["main_claim"],
        "claim_boundaries": claim["required_boundaries"],
        "primary_results": "tables/primary_results.json",
        "mechanism_and_ablation": "tables/mechanism_and_ablation.json",
        "negative_results": "tables/negative_results.json",
        "efficiency": "tables/efficiency.json",
        "main_claim_tables": "tables/main_claim_tables.json",
        "trinity_component_gap": "tables/trinity_component_gap.json",
        "pareto": "figures/chip_token_flow_pareto_v1.svg + .pdf",
        "manifest": "reproducibility_manifest.json",
        "required_reporting": [
            "Report per-seed and seed-mean results.",
            "Report paired case intervals including intervals that cross zero.",
            "Report that same-protocol Trinity d192 remains stronger.",
            "Report structural, tail, and complexity-heterogeneity diagnostics.",
            "Report the equal-quality NFE/refiner budgets and their seed scope.",
            "Report that cost gains concentrate in the structural-failure tail.",
            "Report the non-monotonic V_rel extreme tail and the 50-step deficit.",
            "Report the structural/geometric Trinity decomposition and source residual.",
            "Preserve the metric demotion ledger.",
            "Report legacy and three-seed Trinity anchors with the reproducibility warning.",
            "Do not claim FloorSet SOTA or an informative invariant global state.",
        ],
        "claim_metrics": claim_metrics,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write(args.out_dir / "tables/primary_results.json", primary)
    write(
        args.out_dir / "tables/mechanism_and_ablation.json",
        {
            "parity_gap": parity,
            "graph_mechanism": mechanism,
            "claim_gate": claim,
            "token_flow_mechanism": flow_mechanism,
            "metric_triage": triage,
            "tail_budget_mechanism": tail_budget,
            "trinity_gap_decomposition": gap_decomposition,
            "cpu_bridge": bridge,
        },
    )
    write(args.out_dir / "tables/negative_results.json", negative)
    efficiency["quality_budget"] = budget
    write(args.out_dir / "tables/efficiency.json", efficiency)
    write(args.out_dir / "tables/main_claim_tables.json", claim["main_tables"])
    write(
        args.out_dir / "tables/trinity_component_gap.json",
        {
            **trinity_gap,
            "legacy_anchor": trinity_gap["legacy_anchor"],
            "reproducible_replication": trinity_gap[
                "reproducible_replication"
            ],
            "pareto": pareto,
            "decomposition_v1": gap_decomposition,
        },
    )
    write(args.out_dir / "tables/metric_triage.json", triage)
    write(
        args.out_dir / "tables/tail_budget_mechanism.json",
        tail_budget,
    )
    write(
        args.out_dir / "tables/trinity_gap_decomposition.json",
        gap_decomposition,
    )
    figures_dir = args.out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    for suffix in (".svg", ".pdf"):
        shutil.copy2(
            ROOT / f"outputs/figures/chip_token_flow_pareto_v1{suffix}",
            figures_dir / f"chip_token_flow_pareto_v1{suffix}",
        )
        shutil.copy2(
            ROOT / f"paper/figures/chip_tail_budget_mechanism{suffix}",
            figures_dir / f"chip_tail_budget_mechanism{suffix}",
        )
        shutil.copy2(
            ROOT
            / f"paper/figures/chip_trinity_gap_decomposition{suffix}",
            figures_dir / f"chip_trinity_gap_decomposition{suffix}",
        )
    for relative in EVIDENCE:
        target = args.out_dir / "evidence" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    for relative in dict.fromkeys(SOURCE_FILES):
        target = args.out_dir / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    markdown_tables = (
        "paper/tables/chip_metric_triage.md",
        "paper/tables/chip_tail_budget_mechanism.md",
        "paper/tables/chip_trinity_gap_decomposition.md",
    )
    for relative in markdown_tables:
        target = args.out_dir / "tables" / Path(relative).name
        shutil.copy2(ROOT / relative, target)
    manifest.update(
        {
            "hash_algorithm": "sha256",
            "missing_count": 0,
            "evidence_file_count": len(EVIDENCE),
            "source_file_count": len(dict.fromkeys(SOURCE_FILES)),
            "archive": "outputs/cvpr_package_v13_chips.tar.gz",
        }
    )
    write(args.out_dir / "reproducibility_manifest.json", manifest)
    write(args.out_dir / "submission_checklist.json", checklist)
    manifest["file_count"] = len(
        [path for path in args.out_dir.rglob("*") if path.is_file()]
    )
    write(args.out_dir / "reproducibility_manifest.json", manifest)
    archive_path = ROOT / "outputs/cvpr_package_v13_chips.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(args.out_dir, arcname=args.out_dir.name)
    (ROOT / "outputs/cvpr_package_v13_chips.tar.gz.sha256").write_text(
        f"{digest(archive_path)}  {archive_path.relative_to(ROOT)!s}\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": checklist["status"], **checks}, indent=2))


if __name__ == "__main__":
    main()
