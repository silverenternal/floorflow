"""Audit matched availability and claim safety for residential baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from floorflow.evaluation.e2e_benchmark_contract import (
    load_benchmark_config,
    validate_benchmark_config,
    validate_claim_rows,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/residential_e2e_benchmark_v1.json"
DEFAULT_REPORT = ROOT / "outputs/reports/residential_e2e_baseline_audit_v1.json"
DEFAULT_DOC = ROOT / "docs/residential_e2e_baseline_availability_v1.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    parser.add_argument("--skip-online", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _path_status(relative: str) -> dict:
    path = ROOT / relative
    row = {
        "path": relative,
        "exists": path.exists(),
    }
    if path.is_file():
        row["bytes"] = path.stat().st_size
        row["sha256"] = _sha256(path)
    return row


def _read(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def _remote_head(path: str) -> dict:
    result = subprocess.run(
        ["git", "-C", str(ROOT / path), "remote", "get-url", "origin"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return {"available": False, "error": result.stderr.strip()}
    remote = result.stdout.strip()
    head = subprocess.run(
        ["git", "ls-remote", remote, "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "available": head.returncode == 0,
        "remote": remote,
        "remote_head": (
            head.stdout.split()[0] if head.stdout.strip() else None
        ),
        "error": head.stderr.strip() or None,
    }


def _local_commit(path: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(ROOT / path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _code_files(path: str) -> list[str]:
    root = ROOT / path
    suffixes = {".py", ".ipynb", ".pt", ".pth", ".ckpt"}
    return sorted(
        str(item.relative_to(root))
        for item in root.rglob("*")
        if item.is_file()
        and ".git" not in item.parts
        and item.suffix.lower() in suffixes
    )


def _method_rows(config: dict) -> tuple[list[dict], list[dict]]:
    availability = []
    claims = []
    for method, spec in config["methods"].items():
        availability.append(
            {
                "method": method,
                "role": spec["role"],
                "availability": spec["availability"],
                "matched_status": spec["matched_status"],
                "redistribution": spec["redistribution"],
                "claim_enabled": bool(spec["claim_enabled"]),
                "task_tracks": list(spec["task_tracks"]),
            }
        )
        for track in spec["task_tracks"]:
            if method == "FloorFlow":
                continue
            if spec["availability"] in {
                "unavailable",
                "repository_readme_only",
                "weights_only_inference_blocked",
            }:
                state = "unavailable"
            elif spec["matched_status"] == "not_matched":
                state = "context_only"
            else:
                state = "not_comparable"
            claims.append(
                {
                    "method": method,
                    "track": track,
                    "claim_state": state,
                    "executable": spec["availability"]
                    in {"executable", "local_audit_only"},
                    "same_scorer": False,
                    "reason": _claim_reason(method, track, spec),
                }
            )
    return availability, claims


def _claim_reason(method: str, track: str, spec: dict) -> str:
    if method == "Raster2Seq":
        return (
            "Official CubiCasa reconstruction anchor is reproduced, but no "
            "matched FloorFlow reconstruction run shares its 509-case contract."
        )
    if method == "CE2EPlan":
        return (
            "Official RPLAN generation anchor is reproducible locally, but no "
            "matched FloorFlow RPLAN model or shared scorer exists."
        )
    if method == "HouseMind":
        return (
            "Weights are available, but official tokenizer/serialization/"
            "inference/post-processing contracts remain unreleased."
        )
    if method == "TLC-Plan":
        return (
            "The public repository currently exposes README/assets without "
            "executable training or inference code."
        )
    if method == "FMLM":
        return "No executable code or checkpoint is advertised."
    if method == "RenoLayout":
        return "No matched executable renovation inference package is available."
    return f"{method} has no matched {track} row."


def _markdown(report: dict) -> str:
    lines = [
        "# Residential E2E Baseline Availability v1",
        "",
        f"Status: `{report['status']}`",
        "",
        "| Method | Track | Claim state | Reason |",
        "|---|---|---|---|",
    ]
    for row in report["claim_rows"]:
        lines.append(
            f"| {row['method']} | {row['track']} | {row['claim_state']} | "
            f"{row['reason']} |"
        )
    lines.extend(
        [
            "",
            (
                "Only `matched` rows with the same split, conditioning, "
                "preprocessing, output contract, and scorer can enter a leaderboard."
            ),
            "",
            (
                "External SOTA rows enabled at this gate: "
                f"`{len(report['external_sota_enabled_rows'])}`."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    config = load_benchmark_config(args.config)
    errors = validate_benchmark_config(config)
    availability, claim_rows = _method_rows(config)
    errors.extend(validate_claim_rows(config, claim_rows))
    paths = {
        name: _path_status(relative)
        for name, relative in config["authoritative_paths"].items()
    }
    missing = [name for name, row in paths.items() if not row["exists"]]
    source_reports = {
        "raster2seq": _read(
            config["authoritative_paths"]["raster2seq_official_anchor"]
        ),
        "ce2eplan": _read(
            config["authoritative_paths"]["ce2eplan_anchor"]
        ),
        "housemind": _read(
            config["authoritative_paths"]["housemind_status"]
        ),
        "tlc_plan": _read(
            config["authoritative_paths"]["contemporary_baseline_audit"]
        ),
    }
    tlc_code = _code_files("third_party/TLC-Plan")
    online = {}
    if not args.skip_online:
        online = {
            "TLC-Plan": _remote_head("third_party/TLC-Plan"),
            "Raster2Seq": _remote_head("third_party/Raster2Seq"),
        }
    source_summary = {
        "Raster2Seq": {
            "local_commit": _local_commit("third_party/Raster2Seq"),
            "code_available": bool(_code_files("third_party/Raster2Seq")),
            "official_anchor_status": source_reports["raster2seq"].get("status"),
            "official_room_f1": source_reports["raster2seq"]
            .get("official_anchor", {})
            .get("metrics", {})
            .get("room_f1"),
            "matched_floorflow": False,
        },
        "CE2EPlan": {
            "status": source_reports["ce2eplan"].get("status"),
            "fid": source_reports["ce2eplan"]
            .get("reproduced", {})
            .get("FID"),
            "ged": source_reports["ce2eplan"]
            .get("reproduced", {})
            .get("GED_mean"),
            "license_found": source_reports["ce2eplan"]
            .get("source_availability", {})
            .get("license_found"),
            "matched_floorflow": False,
        },
        "HouseMind": {
            "status": source_reports["housemind"].get("status"),
            "code_status": source_reports["housemind"]
            .get("public_artifacts", {})
            .get("code_status"),
            "matched_editing": False,
        },
        "TLC-Plan": {
            "local_commit": _local_commit("third_party/TLC-Plan"),
            "executable_code_files": tlc_code,
            "readme_only": len(tlc_code) == 0,
        },
        "FMLM": {
            "available": next(
                row
                for row in source_reports["tlc_plan"][
                    "unavailable_remote_baselines"
                ].values()
                if row.get("citation", "").startswith("CVPR 2026")
            ),
        },
        "RenoLayout": {
            "availability": config["methods"]["RenoLayout"]["availability"],
            "matched_editing": False,
        },
    }
    external_sota = [
        row
        for row in claim_rows
        if row["claim_state"] == "matched"
        and row["method"] != "FloorFlow"
    ]
    if missing:
        errors.append(f"missing authoritative artifacts: {missing}")
    if external_sota:
        errors.append("E2E-01 unexpectedly enabled an external SOTA row")
    report = {
        "schema_version": "residential-e2e-baseline-audit-v1",
        "status": "passed" if not errors else "failed",
        "checked_at_utc": datetime.now(UTC).isoformat(),
        "scope": "residential_floorplan_only",
        "config": str(args.config.relative_to(ROOT)),
        "authoritative_paths": paths,
        "method_availability": availability,
        "claim_rows": claim_rows,
        "external_sota_enabled_rows": external_sota,
        "source_summary": source_summary,
        "online_snapshot": online,
        "checks": {
            "config_valid": not validate_benchmark_config(config),
            "claim_rows_valid": not validate_claim_rows(config, claim_rows),
            "authoritative_artifacts_present": not missing,
            "no_unmatched_external_sota": not external_sota,
            "raster2seq_reconstruction_only": all(
                row["track"] == "cubicasa_reconstruction"
                for row in claim_rows
                if row["method"] == "Raster2Seq"
            ),
            "ce2eplan_generation_only": all(
                row["track"] == "rplan_generation"
                for row in claim_rows
                if row["method"] == "CE2EPlan"
            ),
            "house_mind_unavailable": all(
                row["claim_state"] == "unavailable"
                for row in claim_rows
                if row["method"] == "HouseMind"
            ),
        },
        "errors": errors,
        "next_gate": (
            "E2E-02 may build model-origin candidate data only after preserving "
            "the no-unmatched-SOTA state."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.doc.parent.mkdir(parents=True, exist_ok=True)
    args.doc.write_text(_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": report["status"],
                "checks": report["checks"],
                "errors": errors,
            },
            indent=2,
        )
    )
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
