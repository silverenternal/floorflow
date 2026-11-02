#!/usr/bin/env python3
"""Freeze executable external SOTA baselines and their claim boundaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/sota_official_baselines_v2.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/official_baseline_execution_v2.json",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_json_tree(root: Path) -> dict[str, Any]:
    files = sorted(root.glob("*.json"))
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return {"files": len(files), "manifest_sha256": digest.hexdigest()}


def baseline(config: dict[str, Any], identifier: str) -> dict[str, Any]:
    return next(item for item in config["baselines"] if item["id"] == identifier)


def audit_ce2eplan(
    config: dict[str, Any],
    report: dict[str, Any],
    availability: dict[str, Any],
) -> dict[str, Any]:
    spec = baseline(config, "ce2eplan")
    artifact = spec["artifact"]
    data = read_json(ROOT / ".data/external/CE2EPlan/dataset/dataset_json/data_test.json")
    raw = read_json(ROOT / artifact["raw_predictions"])
    post = read_json(ROOT / artifact["postprocessed_predictions"])
    new_root = ROOT / artifact["new_output_root"]
    new_root.mkdir(parents=True, exist_ok=True)
    copied = new_root / "predictions.json"
    if not copied.exists() or sha256_file(copied) != sha256_file(
        ROOT / artifact["postprocessed_predictions"]
    ):
        shutil.copyfile(ROOT / artifact["postprocessed_predictions"], copied)

    published = spec["published_anchor"]
    reproduced = report["reproduced"]
    tolerance = config["tolerance"]
    ratios = reproduced["stat_ratio"]
    checks = {
        "official_case_yield": len(post) == spec["dataset"]["cases"],
        "raw_generated_case_count": len(raw) == len(post),
        "gt_case_count_matches": len(data) == len(post),
        "unique_generated_names": len({row["name"] for row in post}) == len(post),
        "postprocessing_applied": all(
            bool(row.get("adjacencies_aligned")) for row in post
        ),
        "fid_within_absolute_tolerance": abs(
            reproduced["FID"] - published["FID"]
        )
        <= tolerance["ce2eplan_fid_absolute"],
        "paper_statistics_within_tolerance": all(
            abs(ratios[key] - published[anchor]) <= tolerance["ce2eplan_stat_ratio_absolute"]
            for key, anchor in (
                ("nr", "Nravg"),
                ("cl", "Clavg"),
                ("cr", "Cravg"),
                ("al", "Alavg"),
                ("ab", "Abavg"),
                ("ao", "Aoavg"),
            )
        ),
        "source_commit_pinned": availability["code"]["commit"] == spec["commit"],
        "license_boundary_recorded": (
            availability["code"]["license_file_found"] is False
            and spec["redistribution_allowed"] is False
        ),
    }
    provenance = {
        "method": "CE2EPlan",
        "track": spec["track"],
        "commit": spec["commit"],
        "license": spec["license"],
        "official_command": spec["official_command"],
        "dataset": spec["dataset"],
        "weights": spec["weights"],
        "published_anchor": published,
        "reproduced": {
            "FID": reproduced["FID"],
            "GED_independent_evaluator": reproduced["GED_mean"],
            "stat_ratio": ratios,
        },
        "output": {
            "predictions": str(copied.relative_to(ROOT)),
            "predictions_sha256": sha256_file(copied),
        },
        "matched_comparison": spec["matched_comparison"],
        "claim_boundary": spec["claim_boundary"],
    }
    write_json(new_root / "provenance.json", provenance)
    write_json(new_root / "metrics.json", report)
    return {
        "id": spec["id"],
        "status": "verified_anchor_matched_floorflow_blocked",
        "cases": len(post),
        "published": published,
        "reproduced": provenance["reproduced"],
        "checks": checks,
        "passed": all(checks.values()),
        "matched_comparison": spec["matched_comparison"],
        "claim_boundary": spec["claim_boundary"],
        "output_root": artifact["new_output_root"],
    }


def audit_raster2seq(
    config: dict[str, Any],
    metrics_report: dict[str, Any],
    anchor_report: dict[str, Any],
) -> dict[str, Any]:
    spec = baseline(config, "raster2seq")
    artifact = spec["artifact"]
    metrics = metrics_report["metrics"]
    published = spec["published_anchor"]
    tolerance = config["tolerance"]
    new_root = ROOT / artifact["new_output_root"]
    new_root.mkdir(parents=True, exist_ok=True)
    json_tree = hash_json_tree(ROOT / artifact["predictions"])
    result_tree = hash_json_tree(ROOT / artifact["results"])
    case_audit = anchor_report["case_unit_audit"]
    checks = {
        "official_case_yield": metrics_report["cases"] == spec["dataset"]["cases"],
        "prediction_file_count": json_tree["files"] == spec["dataset"]["cases"],
        "result_file_count": result_tree["files"] == spec["dataset"]["cases"],
        "room_f1_within_absolute_tolerance": abs(
            metrics["room_f1"] - published["room_f1"]
        )
        <= tolerance["raster2seq_room_f1_absolute"],
        "same_command_recorded": metrics_report["command"]
        == spec["official_command"],
        "checkpoint_pinned": metrics_report["checkpoint"]["sha256"]
        == spec["weights"]["checkpoint_sha256"],
        "official_case_unit_audited": (
            case_audit["official_annotation_cases"]
            == spec["case_unit"]["official_cases"]
        ),
        "matched_case_unit_blocked_not_hidden": (
            case_audit["bijection_possible_for_all_503_project_floors"]
            is spec["case_unit"]["one_to_one_bijection_available"]
        ),
    }
    provenance = {
        "method": "Raster2Seq",
        "track": spec["track"],
        "commit": spec["commit"],
        "license": spec["license"],
        "official_command": spec["official_command"],
        "dataset": spec["dataset"],
        "weights": spec["weights"],
        "compatibility_patch": spec["local_compatibility_patch"],
        "published_anchor": published,
        "reproduced": metrics,
        "output_hashes": {
            "predictions": json_tree,
            "results": result_tree,
            "metrics_report": sha256_file(ROOT / artifact["metrics_report"]),
            "anchor_report": sha256_file(ROOT / artifact["anchor_report"]),
        },
        "case_unit_audit": case_audit,
        "matched_comparison": spec["matched_comparison"],
        "claim_boundary": spec["claim_boundary"],
    }
    write_json(new_root / "provenance.json", provenance)
    write_json(new_root / "metrics.json", metrics_report)
    write_json(new_root / "case_unit_audit.json", case_audit)
    return {
        "id": spec["id"],
        "status": "verified_anchor_matched_floorflow_blocked",
        "cases": metrics_report["cases"],
        "published": published,
        "reproduced": metrics,
        "checks": checks,
        "passed": all(checks.values()),
        "matched_comparison": spec["matched_comparison"],
        "claim_boundary": spec["claim_boundary"],
        "output_root": artifact["new_output_root"],
    }


def audit_blocked(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for identifier in ("housemind", "tlc_plan", "fmlm", "renolayout"):
        spec = baseline(config, identifier)
        row = {
            "id": spec["id"],
            "method": spec["method"],
            "track": spec["track"],
            "status": spec["status"],
            "eligible_for_main_table": False,
            "matched_comparison": spec["matched_comparison"],
        }
        if identifier == "housemind":
            partial = read_json(ROOT / spec["partial_reproduction"]["report"])
            row["partial_reproduction"] = {
                "status": partial["status"],
                "cases": partial["cases"],
                "outputs_are_floorplans": False,
                "missing_contract": spec["missing_contract"],
            }
            row["checks"] = {
                "weights_hash_recorded": bool(spec["weights"]["omni_model_sha256"]),
                "token_probe_completed": partial["status"]
                == "partial_reproduction_complete_full_inference_blocked",
                "metrics_not_fabricated": spec["partial_reproduction"][
                    "eligible_for_metrics"
                ]
                is False,
            }
        else:
            row["checks"] = {"executable_contract_absent": True}
        row["passed"] = all(row["checks"].values())
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    ce_report = read_json(ROOT / "outputs/reports/ce2eplan_official_anchor_v1.json")
    ce_availability = read_json(ROOT / "outputs/reports/ce2eplan_availability_v1.json")
    raster_metrics = read_json(
        ROOT / "outputs/reports/fp_raster2seq_official_eval_metrics_v1.json"
    )
    raster_anchor = read_json(
        ROOT / "outputs/reports/fp_raster2seq_official_anchor_v1.json"
    )

    rows = [
        audit_ce2eplan(config, ce_report, ce_availability),
        audit_raster2seq(config, raster_metrics, raster_anchor),
        *audit_blocked(config),
    ]
    executable = [row for row in rows if row["status"].startswith("verified")]
    report = {
        "version": config["version"],
        "scope": config["scope"],
        "execution_device": config["execution_device"],
        "oracle_control_boundary": config["oracle_control_boundary"],
        "summary": {
            "executable_verified": sum(row["passed"] for row in executable),
            "executable_expected": len(executable),
            "blocked_or_absent_audited": sum(
                row["passed"] for row in rows if row not in executable
            ),
            "matched_floorflow_rows_enabled": 0,
            "sota_claim_enabled": False,
        },
        "baselines": rows,
        "decision": (
            "Freeze CE2EPlan and Raster2Seq as reproducible anchors. "
            "Do not enter either in a FloorFlow leaderboard until a matched "
            "FloorFlow run shares the same split, preprocessing, case unit, "
            "post-processing, and evaluator."
        ),
        "next_task": "SOTA-02",
    }
    checks = {
        "both_executable_anchors_verified": all(row["passed"] for row in executable),
        "house_mind_partial_only": next(
            row for row in rows if row["id"] == "housemind"
        )["eligible_for_main_table"]
        is False,
        "no_matched_claim_enabled": report["summary"]["matched_floorflow_rows_enabled"]
        == 0,
        "oracle_control_kept_diagnostic": "deployable reconstruction"
        in config["oracle_control_boundary"]["forbidden_use"],
    }
    report["checks"] = checks
    report["gate"] = {
        "passed": all(checks.values()),
        "absolute": "Two official executables reproduce their public anchors within predeclared tolerances.",
        "matched": "No blocked method enters a direct row and no unmatched FloorFlow comparison is enabled.",
    }
    write_json(args.output, report)
    print(json.dumps({"status": report["gate"], "summary": report["summary"]}, indent=2))


__all__ = ["main"]


if __name__ == "__main__":
    main()
