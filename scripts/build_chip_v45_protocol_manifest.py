#!/usr/bin/env python3
"""Freeze the v45 chip hard-gate inputs and reuse audit before experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/chip_hard_gates_v45.json",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=ROOT / "outputs/reports/chip_v45_protocol_manifest.json",
    )
    parser.add_argument(
        "--reuse-output",
        type=Path,
        default=ROOT / "outputs/reports/chip_v45_reuse_audit.json",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(ROOT)),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def unique_paths(values: list[str | Path]) -> list[Path]:
    return sorted(
        {
            (value if isinstance(value, Path) else ROOT / value).resolve()
            for value in values
        }
    )


def cache_arrays(cache: Path) -> list[Path]:
    paths = sorted(cache.glob("*.npy"), key=lambda path: path.name)
    if not paths:
        raise FileNotFoundError(f"cache contains no .npy arrays: {cache}")
    return paths


def collect_reuse_sources(payload: dict[str, Any]) -> list[Path]:
    paths: list[str | Path] = []
    for key, value in payload.items():
        if key == "known_gaps_requiring_small_extensions":
            continue
        if isinstance(value, list):
            paths.extend(
                item
                for item in value
                if isinstance(item, str)
                and item.startswith(("scripts/", "src/", "third_party/", "outputs/"))
            )
    return unique_paths(paths)


def python_size_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(ROOT)),
        "lines": len(path.read_text(encoding="utf-8").splitlines()),
    }


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    todo = read_json(ROOT / "chipstodo.json")
    screen_path = ROOT / config["dev"]["screen_manifest"]
    screen = read_json(screen_path)

    checkpoints = unique_paths(
        list(config["candidate_checkpoints"].values())
        + list(config["matched_control_checkpoints"].values())
    )
    caches = unique_paths([config["dev"]["cache"], config["official"]["cache"]])
    baseline_paths = unique_paths(config["baseline_evidence"])
    required = unique_paths(
        [
            args.config,
            ROOT / "chipstodo.json",
            screen_path,
            *checkpoints,
            *baseline_paths,
        ]
    )
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"v45 protocol inputs are missing: {missing}")

    cache_records = {
        str(cache.relative_to(ROOT)): {
            "metadata": file_record(cache / "metadata.json"),
            "arrays": [file_record(path) for path in cache_arrays(cache)],
        }
        for cache in caches
    }
    selected_indices = [int(index) for index in screen["indices"]]
    selected_counts = [int(row["block_count"]) for row in screen["rows"]]
    checks = {
        "three_candidate_seeds": sorted(config["candidate_checkpoints"]) == ["19", "20", "21"],
        "three_control_seeds": sorted(config["matched_control_checkpoints"]) == ["19", "20", "21"],
        "matched_budget_equal": (
            config["matched_refiner_steps"] == config["saturation_refiner_steps"][3]
        ),
        "screen_has_expected_cases": len(selected_indices) == config["dev"]["screen_cases"],
        "screen_indices_unique": len(set(selected_indices)) == len(selected_indices),
        "screen_counts_unique": len(set(selected_counts)) == len(selected_counts),
        "screen_salt_matches": screen["salt"] == config["dev"]["screen_salt"],
        "official_selection_disabled": not config["selection_policy"][
            "official_outcomes_for_selection"
        ],
        "same_budget_required": config["selection_policy"][
            "same_budget_candidate_and_control"
        ],
        "primary_nfe_matches": config["primary_nfe"] == 4,
        "projector_matches": config["projector_iterations"] == 8,
        "deterministic_glop": config["legalizer"]["glop_threads"] == 1,
    }
    protocol = {
        "version": "chip-v45-protocol-manifest-v1",
        "status": "pass" if all(checks.values()) else "fail",
        "config": file_record(args.config),
        "todo": file_record(ROOT / "chipstodo.json"),
        "screen": {
            **file_record(screen_path),
            "cases": len(selected_indices),
            "indices": selected_indices,
            "block_counts": selected_counts,
        },
        "checkpoints": [file_record(path) for path in checkpoints],
        "caches": cache_records,
        "baseline_evidence": [file_record(path) for path in baseline_paths],
        "comparison": {
            "primary_nfe": config["primary_nfe"],
            "matched_refiner_steps": config["matched_refiner_steps"],
            "projector_iterations": config["projector_iterations"],
            "use_ema": config["use_ema"],
            "legalizer": config["legalizer"],
            "refiner": config["refiner"],
        },
        "checks": checks,
        "selection_statement": (
            "The screen, checkpoints, cache arrays and comparison budget were frozen "
            "without reading any v45 candidate outcome."
        ),
    }

    reuse = todo["reuse_audit"]
    known_gaps = reuse["known_gaps_requiring_small_extensions"]
    reuse_sources = collect_reuse_sources(reuse)
    source_missing = [
        str(path.relative_to(ROOT)) for path in reuse_sources if not path.exists()
    ]
    python_sources = [
        python_size_record(path)
        for path in reuse_sources
        if path.suffix == ".py"
        and path.relative_to(ROOT).parts[0] in {"scripts", "src", "tests"}
    ]
    oversized = [
        row
        for row in python_sources
        if row["lines"] >= 600
    ]
    reuse_report = {
        "version": "chip-v45-reuse-audit-v1",
        "status": "pass" if not source_missing and not oversized else "fail",
        "policy": reuse["rule"],
        "known_gaps": known_gaps,
        "source_records": [
            {
                **file_record(path),
                "kind": "directory" if path.is_dir() else "file",
            }
            if path.is_file()
            else {
                "path": str(path.relative_to(ROOT)),
                "kind": "directory",
            }
            for path in reuse_sources
        ],
        "python_sources": python_sources,
        "missing_sources": source_missing,
        "oversized_python_sources": oversized,
        "checks": {
            "all_reuse_sources_exist": not source_missing,
            "all_reuse_python_files_below_600_lines": not oversized,
            "new_terms_are_optional": "optional term" in " ".join(known_gaps),
            "existing_orfs_metrics_reused": any(
                path.name == "genMetrics.py" for path in reuse_sources
            ),
        },
    }
    if not all(reuse_report["checks"].values()):
        raise RuntimeError(f"reuse audit failed: {reuse_report['checks']}")
    if protocol["status"] != "pass":
        raise RuntimeError(f"protocol checks failed: {checks}")

    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.reuse_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(
        json.dumps(protocol, indent=2) + "\n",
        encoding="utf-8",
    )
    args.reuse_output.write_text(
        json.dumps(reuse_report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "manifest": str(args.manifest_output.relative_to(ROOT)),
                "reuse": str(args.reuse_output.relative_to(ROOT)),
                "screen_cases": len(selected_indices),
                "checkpoints": len(checkpoints),
                "reuse_sources": len(reuse_sources),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
