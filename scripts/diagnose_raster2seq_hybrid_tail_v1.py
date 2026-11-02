#!/usr/bin/env python3
"""Diagnose the structural tail of the Raster2Seq-proposal hybrid."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=(
            ROOT
            / "outputs/reports/raster2seq_floorflow_audit_val503_final_v5.json"
        ),
    )
    parser.add_argument(
        "--baseline-report",
        type=Path,
        default=(
            ROOT
            / "outputs/reports/raster2seq_floorflow_audit_val503_final_v4.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "outputs/reports/raster2seq_hybrid_tail_diagnostic_v1.json"
        ),
    )
    return parser.parse_args()


def _failure_signature(row: dict) -> str:
    return ";".join(sorted(row["plan_failure_codes"]))


def _wall_continuity_bin(value: float) -> str:
    if value >= 0.95:
        return ">=0.95"
    if value >= 0.90:
        return "0.90-0.95"
    if value >= 0.50:
        return "0.50-0.90"
    return "<0.50"


def _display_path(path: Path) -> str:
    resolved = path if path.is_absolute() else ROOT / path
    return str(resolved.resolve().relative_to(ROOT))


def _row_failure(
    row: dict,
    *,
    previously_failed: set[int],
    previously_passed: set[int],
) -> dict:
    image_id = int(row["image_id"])
    if image_id in previously_failed:
        transition = "persistent"
    elif image_id in previously_passed:
        transition = "new"
    else:
        transition = "not_in_baseline"
    return {
        "image_id": image_id,
        "transition": transition,
        "footprint_components": int(row["footprint_component_count"]),
        "exportable_footprint_components": int(
            row["exportable_footprint_component_count"]
        ),
        "room_union_components": int(row["room_union_component_count"]),
        "wall_count": int(row["wall_count"]),
        "wall_continuity_rate": float(row["wall_continuity_rate"]),
        "wall_continuity_bin": _wall_continuity_bin(
            float(row["wall_continuity_rate"])
        ),
        "dangling_endpoint_rate": float(row["dangling_endpoint_rate"]),
        "hole_count": int(row["hole_count"]),
        "room_face_retention": float(row["room_face_retention"]),
        "functional_watertight": float(row["functional_watertight"]),
        "failure_codes": sorted(row["plan_failure_codes"]),
    }


def _group_counts(rows: list[dict], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[key]) for row in rows).items()))


def main() -> None:
    args = parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    rows = report["case_rows"]
    failures = [row for row in rows if not row["plan_plausible"]]
    baseline = json.loads(args.baseline_report.read_text(encoding="utf-8"))
    previously_failed = {
        int(row["image_id"])
        for row in baseline["case_rows"]
        if not row["plan_plausible"]
    }
    previously_passed = {
        int(row["image_id"])
        for row in baseline["case_rows"]
        if row["plan_plausible"]
    }
    failed_rows = [
        _row_failure(
            row,
            previously_failed=previously_failed,
            previously_passed=previously_passed,
        )
        for row in failures
    ]
    signatures: dict[str, list[int]] = defaultdict(list)
    for row in failures:
        signatures[_failure_signature(row)].append(int(row["image_id"]))
    target_sets = {
        "isolated_dangling_wall": [
            row["image_id"]
            for row in failed_rows
            if row["footprint_components"] == 1
            and row["failure_codes"] == ["dangling_endpoints", "wall_continuity"]
        ],
        "room_face_collapse": [
            row["image_id"]
            for row in failed_rows
            if "room_face_collapse" in row["failure_codes"]
        ],
        "multi_component_wall_continuity": [
            row["image_id"]
            for row in failed_rows
            if row["footprint_components"] > 1
            and "wall_continuity" in row["failure_codes"]
        ],
        "multi_component_without_wall_continuity": [
            row["image_id"]
            for row in failed_rows
            if row["footprint_components"] > 1
            and "wall_continuity" not in row["failure_codes"]
        ],
    }
    diagnostic = {
        "schema_version": "raster2seq-hybrid-tail-diagnostic-v1",
        "scope": "Raster2Seq-proposal plus FloorFlow-legalizer hybrid; not standalone FloorFlow",
        "source_report": _display_path(args.report),
        "baseline_report": _display_path(args.baseline_report),
        "cases": len(rows),
        "failures": len(failures),
        "passes": len(rows) - len(failures),
        "plan_plausibility": float(report["summary"]["plan_plausible"]["mean"]),
        "target_plus_one_required": len(failures) - 1,
        "transition_counts": dict(
            sorted(Counter(row["transition"] for row in failed_rows).items())
        ),
        "failure_signatures": [
            {
                "failure_codes": signature.split(";"),
                "count": len(images),
                "images": images,
            }
            for signature, images in sorted(
                signatures.items(),
                key=lambda item: (-len(item[1]), item[0]),
            )
        ],
        "strata": {
            "footprint_components": _group_counts(
                failed_rows,
                "footprint_components",
            ),
            "wall_continuity": _group_counts(
                failed_rows,
                "wall_continuity_bin",
            ),
            "functional_watertight": _group_counts(
                failed_rows,
                "functional_watertight",
            ),
        },
        "repair_targets": target_sets,
        "fixed_tail_cases": sorted(
            {
                4,
                54,
                107,
                *target_sets["multi_component_wall_continuity"][:5],
            }
        ),
        "failure_rows": sorted(failed_rows, key=lambda row: row["image_id"]),
        "decision": (
            "Use a fixed tail and adjacent control screen. Prioritize the two "
            "isolated dangling-wall cases; do not increase the global gap radius."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(diagnostic, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(diagnostic, indent=2))


if __name__ == "__main__":
    main()
