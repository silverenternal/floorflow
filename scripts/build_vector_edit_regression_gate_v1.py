#!/usr/bin/env python3
"""Verify the vector work does not mutate frozen raster evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "outputs/cvpr_package_fp_v17/manifest.json"
OUTPUT = ROOT / "outputs/reports/vector_edit_regression_gate_v1.json"
KEY_ARTIFACTS = (
    "outputs/reports/fevp_case_rows_v2.json",
    "outputs/reports/fevp_fiveseed_stability_v2.json",
    "outputs/reports/fevp_graph_metrics_v2.json",
    "outputs/reports/fevp_validity_watertight_v2.json",
    "outputs/reports/fp_control_gated_hard_gate_v1.json",
)


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    frozen = manifest["artifacts"]
    rows = {}
    for relative in KEY_ARTIFACTS:
        current = _digest(ROOT / relative)
        expected = frozen[relative]["sha256"]
        rows[relative] = {
            "current_sha256": current,
            "frozen_v17_sha256": expected,
            "unchanged": current == expected,
        }
    stability = json.loads(
        (ROOT / "outputs/reports/fevp_fiveseed_stability_v2.json").read_text(
            encoding="utf-8"
        )
    )
    graph = json.loads(
        (ROOT / "outputs/reports/fevp_graph_metrics_v2.json").read_text(
            encoding="utf-8"
        )
    )
    checks = {
        "frozen_artifacts_unchanged": all(
            row["unchanged"] for row in rows.values()
        ),
        "raster_delta_iou_unchanged": (
            stability["metrics"]["delta_iou"]["mean"] == 0.9870840184952085
        ),
        "raster_delta_mse_unchanged": (
            stability["metrics"]["delta_mse"]["mean"]
            == 0.00021761272324362862
        ),
        "raster_graph_gate_still_true": graph["summary"][
            "typed_ged_all_paired_tests_p_lt_0_05"
        ],
    }
    report = {
        "status": "complete" if all(checks.values()) else "failed",
        "frozen_package": str(MANIFEST.relative_to(ROOT)),
        "artifacts": rows,
        "checks": checks,
        "boundary": (
            "This verifies numerical regression against frozen v17 raster "
            "evidence. Vector modules are additive and do not alter those files."
        ),
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "checks": checks}, indent=2))
    raise SystemExit(0 if report["status"] == "complete" else 1)


__all__ = ["main"]


if __name__ == "__main__":
    main()
