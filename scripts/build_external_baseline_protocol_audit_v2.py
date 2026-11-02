#!/usr/bin/env python3
"""Audit fairness and reproducibility boundaries for external baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matched",
        type=Path,
        default=ROOT / "outputs/reports/fp_edit_detection_matched_external_v2.json",
    )
    parser.add_argument(
        "--official",
        type=Path,
        default=ROOT / "outputs/reports/fp_raster2seq_official_anchor_v1.json",
    )
    parser.add_argument(
        "--ce2e",
        type=Path,
        default=ROOT / "outputs/reports/fevp_ce2eplan_portability_v3.json",
    )
    parser.add_argument(
        "--house",
        type=Path,
        default=ROOT / "outputs/reports/fevp_housemind_portability_gate_v1.json",
    )
    parser.add_argument(
        "--wrapper",
        type=Path,
        default=ROOT / "scripts/evaluate_fp_raster2seq_local_redraw.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/external_baseline_protocol_audit_v2.json",
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=ROOT / "paper/tables/external_baseline_protocol_audit_v2.md",
    )
    return parser.parse_args()


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()
    matched = _read(args.matched)
    official = _read(args.official)
    ce2e = _read(args.ce2e)
    house = _read(args.house)
    report = {
        "status": "complete",
        "raster2seq": {
            "official_anchor_status": official["status"],
            "official_room_f1": official["official_anchor"]["metrics"]["room_f1"],
            "matched_sensitivity_status": matched["status"],
            "matched_scope": matched["scope"],
            "shared_project_preprocessing": matched["checks"][
                "shared_project_preprocessing_sensitivity"
            ],
            "upstream_official_preprocessing_matched": matched["checks"][
                "upstream_official_preprocessing_matched"
            ],
            "wrapper": {
                "path": str(args.wrapper.relative_to(ROOT)),
                "sha256": _sha(args.wrapper),
                "implementation_public": True,
            },
        },
        "ce2eplan": {
            "sampling_method": ce2e["sampling_method"],
            "sampling_seed": ce2e["sampling_seed"],
            "sample_indices_sha256": ce2e["sample_indices_sha256"],
            "sample_count": ce2e["cases"],
            "outside_exact": ce2e["outside_exact"],
        },
        "housemind": {
            "status": house["status"],
            "verified_at": house["verified_at"],
            "official_code_status": house["official_code_status"],
            "source_hashes": {
                name: value.get("sha256")
                for name, value in house["sources"].items()
            },
        },
        "checks": {
            "official_and_shared_preprocessing_separated": (
                official["status"]
                == "official_anchor_complete_matched_identity_blocked"
                and matched["checks"][
                    "upstream_official_preprocessing_matched"
                ]
                is False
            ),
            "wrapper_is_public_and_hashed": (
                args.wrapper.is_file()
                and len(_sha(args.wrapper)) == 64
            ),
            "ce2e_sampling_disclosed": (
                ce2e["sampling_method"] == "np.linspace_over_public_test_order"
                and ce2e["sampling_seed"] is None
                and len(ce2e["sample_indices"]) == ce2e["cases"]
                and len(ce2e["sample_indices_sha256"]) == 64
            ),
            "house_mind_timestamp_and_hashes": (
                house["status"] == "blocked_external"
                and bool(house["verified_at"])
                and all(house["sources"][name].get("sha256") for name in house["sources"])
            ),
        },
        "claims": {
            "allowed": [
                (
                    "Raster2Seq matched rows are a shared-project-preprocessing "
                    "sensitivity analysis, not an official-pipeline reproduction."
                ),
                "The Raster2Seq outside-frozen wrapper is public and reproducible.",
                "CE2EPlan samples are deterministic evenly spaced and index-hashed.",
                "HouseMind unavailability is time-stamped and page-hash backed.",
            ],
            "forbidden": [
                "The matched edit comparison uses official Raster2Seq preprocessing.",
                "HouseMind is failed rather than unreported.",
                "CE2EPlan is a matched editing baseline.",
            ],
        },
    }
    failures = [key for key, value in report["checks"].items() if not value]
    if failures:
        report["status"] = "failed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    table = [
        "# External Baseline Protocol Audit",
        "",
        "| Item | State |",
        "|---|---|",
        f"| Raster2Seq official anchor | {official['status']} |",
        f"| Raster2Seq matched rows | shared project preprocessing sensitivity |",
        f"| Raster2Seq wrapper SHA-256 | `{_sha(args.wrapper)}` |",
        f"| CE2EPlan sampling | {ce2e['sampling_method']} |",
        f"| CE2EPlan sample count | {ce2e['cases']} |",
        f"| HouseMind gate | {house['status']} at {house['verified_at']} |",
        "",
        "Official Raster2Seq evaluation and shared-preprocessing edit sensitivity are separate experiments.",
        "",
    ]
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text("\n".join(table), encoding="utf-8")
    print(json.dumps({"status": report["status"], **report["checks"]}, indent=2))
    raise SystemExit(0 if report["status"] == "complete" else 1)


__all__ = ["main"]


if __name__ == "__main__":
    main()
