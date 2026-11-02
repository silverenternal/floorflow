#!/usr/bin/env python3
"""Materialize per-case chip proxy/canonical measurement payloads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from floorflow.chip_flow.training_runtime import write_json_atomic


def _read_path(payload: dict[str, Any], path: str) -> float:
    value: Any = payload
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(path)
        value = value[part]
    return float(value)


def _oriented(
    value: float,
    direction: str,
    transform: str = "identity",
) -> float:
    result = abs(value) if transform == "absolute" else value
    return -result if direction == "minimize" else result


def _source_rows(source: dict[str, str]) -> dict[int, dict]:
    report = json.loads(Path(source["path"]).read_text(encoding="utf-8"))
    rows = report.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{source['path']} has no per-case rows")
    return {int(row["index"]): row for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/chip_proxy_measurement_sources_v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reports/chip_proxy_measurements_v1.json"),
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    source_rows = {
        source["label"]: _source_rows(source)
        for source in config["sources"]
    }
    common_cases = set.intersection(
        *(set(rows) for rows in source_rows.values())
    )
    if not common_cases:
        raise ValueError("chip proxy sources share no cases")
    checkpoints = [
        f"{source['label']}:{candidate}"
        for source in config["sources"]
        for candidate in config["candidates"]
    ]
    measurements = []
    for spec in config["measurements"]:
        rows = []
        by_checkpoint: dict[str, list[tuple[float, float, float, float]]] = {}
        for case_id in sorted(common_cases):
            for source in config["sources"]:
                case = source_rows[source["label"]][case_id]
                for candidate in config["candidates"]:
                    candidate_row = case[candidate]
                    checkpoint = f"{source['label']}:{candidate}"
                    proxy = _read_path(candidate_row, spec["proxy_path"])
                    canonical = _read_path(
                        candidate_row,
                        spec["canonical_path"],
                    )
                    rows.append(
                        {
                            "seed": int(config["analysis_seed"]),
                            "checkpoint": checkpoint,
                            "case_id": str(case_id),
                            "proxy": proxy,
                            "canonical": canonical,
                        }
                    )
                    by_checkpoint.setdefault(checkpoint, []).append(
                        (
                            _oriented(
                                proxy,
                                spec["proxy_direction"],
                                spec.get("proxy_transform", "identity"),
                            ),
                            _oriented(
                                canonical,
                                spec["canonical_direction"],
                                spec.get("canonical_transform", "identity"),
                            ),
                            proxy,
                            canonical,
                        )
                    )
        baseline_name = config["baseline_checkpoint"]
        baseline_proxy = sum(
            row[0] for row in by_checkpoint[baseline_name]
        ) / len(by_checkpoint[baseline_name])
        baseline_canonical = sum(
            row[1] for row in by_checkpoint[baseline_name]
        ) / len(by_checkpoint[baseline_name])
        baseline_proxy_raw = sum(
            row[2] for row in by_checkpoint[baseline_name]
        ) / len(by_checkpoint[baseline_name])
        baseline_canonical_raw = sum(
            row[3] for row in by_checkpoint[baseline_name]
        ) / len(by_checkpoint[baseline_name])
        sensitivity = []
        for checkpoint, values in by_checkpoint.items():
            if checkpoint == baseline_name:
                continue
            proxy_mean = sum(row[0] for row in values) / len(values)
            canonical_mean = sum(row[1] for row in values) / len(values)
            if proxy_mean < baseline_proxy and canonical_mean < baseline_canonical:
                proxy_raw = sum(row[2] for row in values) / len(values)
                canonical_raw = sum(row[3] for row in values) / len(values)
                sensitivity.append(
                    {
                        "case_id": "cohort_mean",
                        "attack": f"candidate_arm_contrast:{checkpoint}",
                        "proxy_baseline": baseline_proxy_raw,
                        "proxy_attack": proxy_raw,
                        "canonical_baseline": baseline_canonical_raw,
                        "canonical_attack": canonical_raw,
                    }
                )
                break
        measurements.append(
            {
                **spec,
                "baseline_checkpoint": config["baseline_checkpoint"],
                "target_used_at_inference": False,
                "rows": rows,
                "sensitivity": sensitivity,
                "source_kind": "candidate_arm_views",
                "evidence_boundary": (
                    "Six model/legalizer arms stand in for checkpoints; "
                    "sensitivity uses an observed candidate-arm degradation "
                    "contrast, not a synthetic geometry attack."
                ),
            }
        )
    write_json_atomic(
        args.output,
        {
            "version": "chip-proxy-measurements-v1",
            "analysis_seed": int(config["analysis_seed"]),
            "checkpoints": checkpoints,
            "common_cases": len(common_cases),
            "measurements": measurements,
        },
    )
    print(
        json.dumps(
            {
                "cases": len(common_cases),
                "checkpoints": len(checkpoints),
                "measurements": len(measurements),
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
