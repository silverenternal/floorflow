#!/usr/bin/env python3
"""Build the v45 claim gate and reproducible negative-evidence package."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_chip_cvpr_package import digest, python_file_sizes, read, write

GATE_MANIFEST = read(ROOT / "configs/chip_v45_venue_gate_manifest.json")
EVIDENCE = GATE_MANIFEST["evidence"]

SOURCE_FILES = tuple(GATE_MANIFEST["source_files"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pytest-xml",
        type=Path,
        default=ROOT / "outputs/reports/chip_pytest_final_v45.xml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_v45_final_venue_gate.json",
    )
    parser.add_argument(
        "--package-dir",
        type=Path,
        default=ROOT / "outputs/cvpr_package_chips_v45",
    )
    return parser.parse_args()


def pytest_summary(path: Path) -> dict[str, int]:
    root = ET.parse(path).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    if suite is None:
        raise ValueError(f"no testsuite in {path}")
    return {
        key: int(suite.attrib.get(key, 0))
        for key in ("tests", "failures", "errors", "skipped")
    }


def copy_preserving_path(source: Path, target_root: Path) -> None:
    relative = source.relative_to(ROOT)
    target = target_root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def checksums(directory: Path) -> dict[str, str]:
    return {
        str(path.relative_to(directory)): digest(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != "SHA256SUMS"
    }


def main() -> None:
    args = parse_args()
    evidence = {name: read(ROOT / path) for name, path in EVIDENCE.items()}
    tests = pytest_summary(args.pytest_xml)
    sizes = python_file_sizes()
    oversized = {path: lines for path, lines in sizes.items() if lines >= 600}
    matched = evidence["matched_compute"]
    external = evidence["external_holdout"]
    endpoints = evidence["endpoint_certification"]
    adapter = evidence["adapter_preflight"]
    route_screen = evidence["routing_norm_screen"]
    postprocess = evidence["postprocess_screen"]
    candidates = evidence["candidate_registry"]["candidates"]
    quick_only_statuses = {
        "provisional",
        "quick_pass_promote_to_confirmation",
        "provisional_pending_confirmation",
    }
    rejected = {
        item.get("id"): item.get("status")
        for item in evidence["candidate_registry"]["rejected_screens"]
    }
    checks = {
        "matched_compute_rejected": matched["status"] == "rejected",
        "external_holdout_rejected": (
            external["status"] == "reject_incomplete_external_holdout"
        ),
        "external_holdout_incomplete_disclosed": (
            external["completed_pairs"] == 13
            and external["required_complete_pairs"] == 15
            and len(external["failed_cases"]) == 2
        ),
        "completed_have_endpoints": (
            endpoints["runs_certified"] == 26 and not endpoints["missing"]
        ),
        "adapter_preflight_passed": adapter["status"] == "pass",
        "routing_training_rejected": route_screen["status"] == "rejected_5k",
        "postprocess_rejected": postprocess["status"] == "rejected_multicase",
        "m5_screen_passed": (
            evidence["m5_typed_bias_quick"]["status"]
            == "quick_pass_promote_to_confirmation"
        ),
        "m5_confirmation_rejected": (
            evidence["m5_typed_bias_confirmation"]["status"]
            == "rejected_confirmation"
            and evidence["m5_typed_layers_confirmation"]["status"]
            == "rejected_confirmation"
            and evidence["m5_grouping_confirmation"]["status"]
            == "rejected_confirmation"
        ),
        "no_quick_only_candidate": all(
            candidate.get("status") not in quick_only_statuses
            for candidate in candidates
        ),
        "subset_candidate_confirmation_passed": (
            evidence["m5_balanced_stream_quick"]["status"]
            == "quick_pass_promote_to_confirmation"
            and evidence["m5_balanced_stream_confirmation"]["status"]
            == "confirmation_pass_promote_to_full"
        ),
        "balanced_stream_full_rejected": (
            evidence["m5_balanced_stream_full"]["status"]
            == "rejected_full_data"
            and evidence["m5_balanced_stream_full"]["checks"]["cost_improves"]
            and not evidence["m5_balanced_stream_full"]["checks"]["v_rel_improves"]
        ),
        "rejected_full_candidate_preserved": (
            not any(
                item.get("id") == "balanced_stream_sampler"
                for item in evidence["candidate_registry"]["candidates"]
            )
            and any(
                item.get("id") == "balanced_stream_sampler"
                and item.get("status") == "rejected_full_data"
                for item in evidence["candidate_registry"]["rejected_screens"]
            )
        ),
        "a1_boundary_interference_localized": (
            evidence["a1_boundary_probe"]["status"]
            == "post_legalization_boundary_tail_interference_supported"
            and evidence["a1_boundary_probe"]["checks"][
                "raw_boundary_mean_improves"
            ]
            and evidence["a1_boundary_probe"]["checks"][
                "legalized_boundary_mean_worsens"
            ]
        ),
        "a1_postrefiner_projection_rejected": (
            evidence["a1_boundary_quick"]["status"] == "rejected_quick_screen"
            and evidence["a1_boundary_quick"]["checks"]["cost_improves"] is False
            and evidence["a1_boundary_quick"]["checks"]["v_rel_improves"] is False
            and evidence["a1_boundary_quick"]["comparison"][
                "legalized_boundary"
            ]["relative"] < 0.0
        ),
        "a2_soft_boundary_rejected": (
            evidence["a2_soft_boundary_quick"]["status"]
            == "rejected_quick_screen"
            and evidence["a2_soft_boundary_quick"]["checks"][
                "cost_improves"
            ] is False
            and evidence["a2_soft_boundary_quick"]["checks"][
                "v_rel_improves"
            ] is False
        ),
        "a3_boundary_axis_rejected_for_seed_instability": (
            evidence["a3_boundary_axis_quick"]["status"]
            == "quick_pass_promote_to_confirmation"
            and evidence["a3_boundary_axis_confirmation"]["status"]
            == "confirmation_pass_promote_to_full"
            and evidence["a3_seed19_quick"]["status"] == "rejected_quick_screen"
            and not any(
                item.get("id") == "boundary_axis_lock"
                for item in evidence["candidate_registry"]["candidates"]
            )
            and rejected.get("boundary_axis_lock") == "rejected_seed_stability"
        ),
        "a4_corner_boundary_rejected": (
            evidence["a4_corner_boundary_quick"]["status"]
            == "rejected_quick_screen"
            and evidence["a4_corner_boundary_quick"]["checks"][
                "cost_improves"
            ] is False
            and evidence["a4_corner_boundary_quick"]["checks"][
                "v_rel_improves"
            ] is False
            and rejected.get("corner_aware_boundary_lock") == "rejected_quick_screen"
        ),
        "a5_edge_gate_passed_subset_then_rejected_full": (
            evidence["a5_edge_gate_quick"]["status"]
            == "quick_pass_promote_to_confirmation"
            and evidence["a5_edge_gate_confirmation"]["status"]
            == "confirmation_pass_promote_to_full"
            and evidence["a5_edge_gate_full"]["status"] == "rejected_full_data"
            and evidence["a5_edge_gate_full"]["checks"]["cost_improves"]
            and not evidence["a5_edge_gate_full"]["checks"]["v_rel_improves"]
            and rejected.get("netlist_edge_gate") == "rejected_full_data"
        ),
        "a6_delayed_grouping_rejected": (
            evidence["a6_delayed_grouping_confirmation"]["status"]
            == "rejected_confirmation"
            and evidence["a6_delayed_grouping_confirmation"]["checks"][
                "cost_improves"
            ] is False
            and evidence["a6_delayed_grouping_confirmation"]["checks"][
                "v_rel_improves"
            ] is False
            and rejected.get("delayed_grouping_proxy") == "rejected_confirmation"
        ),
        "a7_balanced_edge_factorial_rejected": (
            evidence["a7_balanced_edge_vs_balanced"]["status"]
            == "rejected_quick_screen"
            and evidence["a7_balanced_edge_vs_edge"]["status"]
            == "rejected_quick_screen"
            and rejected.get("balanced_stream_plus_edge_gate")
            == "rejected_quick_factorial"
        ),
        "a8_200k_scaling_rejected_for_seed_instability": (
            evidence["a8_200k_scaling_full"]["status"]
            == "full_pass_promote_to_multiseed"
            and evidence["a8_200k_multiseed"]["status"]
            == "reject_seed_instability"
            and rejected.get("data_scaling_200k") == "rejected_seed_instability"
        ),
        "a9_fixed_data_order_rejected": (
            evidence["a9_fixed_data_order"]["status"]
            == "rejected_confirmation"
            and rejected.get("fixed_data_order_seed21") == "rejected_confirmation"
        ),
        "a10_checkpoint_soup_rejected": (
            evidence["a10_checkpoint_soup"]["status"]
            == "rejected_confirmation"
            and rejected.get("200k_checkpoint_soup") == "rejected_confirmation"
        ),
        "a11_raw_seed21_rejected": (
            evidence["a11_ema_diagnostic"]["status"]
            == "rejected_confirmation"
            and evidence["a11_ema_diagnostic"]["checks"]["cost_improves"] is False
            and evidence["a11_ema_diagnostic"]["checks"]["v_rel_improves"] is False
            and rejected.get("seed21_raw_model") == "rejected_confirmation"
        ),
        "a12_ema_blend_rejected": (
            evidence["a12_ema_blend"]["status"] == "rejected_confirmation"
            and evidence["a12_ema_blend"]["checks"]["cost_improves"] is False
            and evidence["a12_ema_blend"]["checks"]["v_rel_improves"] is False
            and rejected.get("seed21_model_ema_blend") == "rejected_confirmation"
        ),
        "a13_longer_training_rejected": (
            evidence["a13_longer_training"]["status"] == "reject_seed_instability"
            and not evidence["a13_longer_training"]["checks"][
                "all_seeds_improve_both"
            ]
            and rejected.get("200k_longer_training_40k") == "rejected_seed_instability"
        ),
        "a14_structural_flow_rejected": (
            evidence["a14_structural_flow_quick"]["status"]
            == "quick_pass_promote_to_confirmation"
            and evidence["a14_structural_flow_confirmation"]["status"]
            == "rejected_confirmation"
            and evidence["a14_structural_flow_confirmation"]["checks"][
                "cost_improves"
            ]
            and not evidence["a14_structural_flow_confirmation"]["checks"][
                "v_rel_improves"
            ]
            and rejected.get("structural_flow_weight_0p3") == "rejected_confirmation"
        ),
        "a15_energy_guidance_rejected": (
            evidence["a15_energy_guidance_smoke"]["status"] == "pass"
            and evidence["a15_energy_guidance_screen"]["status"]
            == "rejected_confirmation"
            and evidence["a15_energy_guidance_screen"]["checks"][
                "cost_improves"
            ] is False
            and evidence["a15_energy_guidance_screen"]["checks"][
                "v_rel_improves"
            ] is False
            and rejected.get("flow_sampler_energy_guidance_0p02")
            == "rejected_confirmation"
        ),
        "a16_composite_energy_rejected": (
            evidence["a16_composite_energy_smoke"]["status"] == "pass"
            and evidence["a16_composite_energy_screen"]["status"]
            == "rejected_confirmation"
            and rejected.get("composite_sampler_energy_0p01")
            == "rejected_confirmation"
        ),
        "subset_screen_configured": (
            evidence["candidate_registry"]["policy"]["full_data_requires_candidate_registration"]
            and evidence["candidate_registry"]["policy"]["subset_confirmation_required"]
            and evidence["candidate_registry"]["policy"][
                "full_data_requires_same_seed_control"
            ]
        ),
        "pytest_pass": tests["failures"] == 0 and tests["errors"] == 0,
        "file_size_guard": not oversized,
    }
    report = {
        "version": "2026-10-30-chip-v45-final-venue-gate-v2",
        "status": "complete_negative_evidence_package",
        "checks": checks,
        "decision": {
            "matched_compute_sota": "reject",
            "external_routed_ppa": "reject_incomplete_and_negative",
            "external_subgroup_claim": "reject_prerequisite_incomplete",
            "cvpr_ppm_headline": "reject",
            "publishable_cvpr_evidence_ready": False,
            "reason": (
                "The candidate loses the equal-refiner Trinity comparison, the "
                "normalized routing arm fails its legalized screen, proxy-guided "
                "postprocessing fails multicase, the balanced full-data candidate "
                "improves cost but fails V_rel, hard and soft post-refiner projection "
                "are rejected at quick screen, sampler-internal boundary-axis locking "
                "passes seed23 but fails seed19, its corner-aware variant also fails "
                "seed19, the learned edge gate passes quick/confirmation but fails "
                "full-data V_rel, delayed grouping also fails confirmation, and the "
                "balanced-plus-edge factorial fails to beat both single arms; the "
                "200k scaling passes seed23 but fails multiseed stability, fixed "
                "data order and checkpoint soup do not repair it, raw seed21 and "
                "model-EMA blends are worse, 40k continuation remains seed-unstable, "
                "stronger structural flow supervision, routing-only and composite "
                "sampler guidance fail legalized screens, and the external holdout has "
                "13/15 complete pairs with negative aggregate evidence."
            ),
        },
        "key_results": {
            "matched_compute": matched,
            "routing_training": route_screen,
            "postprocess": postprocess,
            "external": external,
            "mechanism_screens": {
                "balanced_stream_full": evidence["m5_balanced_stream_full"],
                "a1_boundary_probe": evidence["a1_boundary_probe"],
                "a1_boundary_quick": evidence["a1_boundary_quick"],
                "a2_soft_boundary_quick": evidence["a2_soft_boundary_quick"],
                "a3_boundary_axis_quick": evidence["a3_boundary_axis_quick"],
                "a3_boundary_axis_confirmation": (
                    evidence["a3_boundary_axis_confirmation"]
                ),
                "a3_seed19_quick": evidence["a3_seed19_quick"],
                "a4_corner_boundary_quick": evidence["a4_corner_boundary_quick"],
                "a5_edge_gate_quick": evidence["a5_edge_gate_quick"],
                "a5_edge_gate_confirmation": evidence["a5_edge_gate_confirmation"],
                "a5_edge_gate_full": evidence["a5_edge_gate_full"],
                "a6_delayed_grouping_confirmation": (
                    evidence["a6_delayed_grouping_confirmation"]
                ),
                "a7_balanced_edge_vs_balanced": (
                    evidence["a7_balanced_edge_vs_balanced"]
                ),
                "a7_balanced_edge_vs_edge": (
                    evidence["a7_balanced_edge_vs_edge"]
                ),
                "a8_200k_scaling_full": evidence["a8_200k_scaling_full"],
                "a8_200k_multiseed": evidence["a8_200k_multiseed"],
                "a9_fixed_data_order": evidence["a9_fixed_data_order"],
                "a10_checkpoint_soup": evidence["a10_checkpoint_soup"],
                "a11_ema_diagnostic": evidence["a11_ema_diagnostic"],
                "a12_ema_blend": evidence["a12_ema_blend"],
                "a13_longer_training": evidence["a13_longer_training"],
                "a14_structural_flow_quick": (
                    evidence["a14_structural_flow_quick"]
                ),
                "a14_structural_flow_confirmation": (
                    evidence["a14_structural_flow_confirmation"]
                ),
                "a15_energy_guidance_smoke": (
                    evidence["a15_energy_guidance_smoke"]
                ),
                "a15_energy_guidance_screen": (
                    evidence["a15_energy_guidance_screen"]
                ),
                "a16_composite_energy_smoke": (
                    evidence["a16_composite_energy_smoke"]
                ),
                "a16_composite_energy_screen": (
                    evidence["a16_composite_energy_screen"]
                ),
            },
            "endpoints": {
                "runs_certified": endpoints["runs_certified"],
                "pairs": len(endpoints["pairs"]),
                "missing": endpoints["missing"],
            },
        },
        "next_mechanism_iteration": {
            "screen_config": "configs/chip_mechanism_screen_v1.json",
            "train_subset": "configs/chip_train_screen_5k_v1.json",
            "candidate_registry": "configs/chip_candidate_registry.json",
            "rule": (
                "Run new mechanisms on the 2k quick and 5k confirmation screens "
                "over the frozen 5k/32-case subsets; only a confirmed candidate "
                "may enter a same-seed matched full-data check."
            ),
        },
        "engineering": {
            "pytest": tests,
            "scoped_ruff": "pass",
            "full_repo_ruff_clean": False,
            "full_repo_ruff_note": (
                "Vendored third_party trees contain pre-existing lint findings."
            ),
            "max_python_file_lines": max(sizes.values()),
            "python_files_600_or_more": oversized,
        },
    }

    package = args.package_dir
    if package.exists():
        shutil.rmtree(package)
    tables = package / "tables"
    source = package / "source"
    tables.mkdir(parents=True)
    source.mkdir(parents=True)
    for name in EVIDENCE:
        write(tables / f"{name}.json", evidence[name])
    for relative in SOURCE_FILES:
        copy_preserving_path(ROOT / relative, source)
    manifest = {
        "version": "2026-10-30-chip-v45-reproducibility-v2",
        "evidence": {
            name: {"path": path, "sha256": digest(ROOT / path)}
            for name, path in EVIDENCE.items()
        },
        "source": {
            path: {"sha256": digest(ROOT / path), "lines": sizes.get(path)}
            for path in SOURCE_FILES
        },
        "pytest": tests,
        "max_python_file_lines": max(sizes.values()),
        "oversized_python_files": oversized,
        "raw_artifact_roots": [
            "outputs/chip_ppa/orfs/results",
            "outputs/reports/orfs/ppa",
            "outputs/reports/chip_def_adapter_v45",
        ],
    }
    write(package / "reproducibility_manifest.json", manifest)
    write(
        package / "submission_checklist.json",
        {
            "scope": "Chip experiments only; no manuscript text.",
            "status": "negative_evidence_package_complete",
            "allowed_claims": [
                "Adapter and endpoint certification for 26 completed routed flows.",
                "Negative matched-compute, routing-arm and postprocess results.",
                (
                    "The balanced full-data sampler improves cost and structural "
                    "metrics but fails the required V_rel gate."
                ),
                (
                    "Post-refiner hard projection improves boundary but fails the "
                    "quick cost and V_rel gate."
                ),
                (
                    "A 0.25 blend toward post-refiner projection also fails the "
                    "quick screen without fixing boundary."
                ),
                (
                    "Sampler-internal boundary-axis locking passes seed23 but "
                    "fails seed19, so it is retained only as a seed-specific "
                    "observation."
                ),
                (
                    "Corner-aware boundary locking still fails seed19 and is not "
                    "promoted."
                ),
                (
                    "The learned netlist edge gate improves subset quick and "
                    "confirmation metrics but fails the full-data V_rel gate."
                ),
                (
                    "The delayed grouping proxy fails confirmation and is rejected."
                ),
                (
                    "Combining balanced sampling with edge gating fails to beat both "
                    "single mechanisms at quick budget."
                ),
                (
                    "200k data scaling improves seed mean and two of three seeds but "
                    "fails the strict three-seed and Trinity-anchor gates."
                ),
                (
                    "Fixed data order and equal-weight checkpoint soup are rejected."
                ),
                (
                    "Raw seed21 and intermediate model-EMA blends are rejected; "
                    "full EMA is the best examined state."
                ),
                (
                    "Extending 200k training to 40k improves mean V_rel but leaves "
                    "cost neutral and fails the all-seed gate."
                ),
                (
                    "Increasing structural trajectory supervision to 0.3 fails the "
                    "confirmation V_rel gate."
                ),
                (
                    "Normalized routing-proxy guidance during NFE4 sampling improves "
                    "p2b but regresses legalized grouping and both primary metrics."
                ),
                (
                    "Composite routing/cluster/legality guidance is rejected and "
                    "remains worse than unguided sampling."
                ),
            ],
            "forbidden_claims": [
                "Matched-compute SOTA.",
                "External routed-PPA improvement.",
                "15-design complete holdout.",
                "Subgroup generalization.",
            ],
        },
    )
    checksum_map = checksums(package)
    (package / "SHA256SUMS").write_text(
        "\n".join(
            f"{digest_value}  {relative}"
            for relative, digest_value in checksum_map.items()
        )
        + "\n",
        encoding="utf-8",
    )
    archive = package.with_suffix(".tar.gz")
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(package, arcname=package.name)
    report["package"] = {
        "directory": str(package.relative_to(ROOT)),
        "archive": str(archive.relative_to(ROOT)),
        "archive_sha256": digest(archive),
        "sha256sums": str((package / "SHA256SUMS").relative_to(ROOT)),
    }
    write(args.output, report)
    print(json.dumps({"status": report["status"], "checks": checks}, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
