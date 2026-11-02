#!/usr/bin/env python3
"""Freeze one matched candidate protocol and audit the Trinity gap."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (19, 20, 21)
METRICS = ("legalized_hard_cost", "legalized_v_rel")
CHECKPOINTS = {
    19: "outputs/chip_hierarchy/v7_bbox_anneal_20k/"
    "typed_hierarchy_bbox_anneal_seed19_20k/checkpoints/final.pt",
    20: "outputs/chip_hierarchy/v10_three_seed/"
    "typed_hierarchy_seed20_20k/checkpoints/final.pt",
    21: "outputs/chip_hierarchy/v10_three_seed/"
    "typed_hierarchy_seed21_20k/checkpoints/final.pt",
}
OLD_CASES = {
    seed: "outputs/chip_hierarchy/v10_three_seed/reproducible_v1/"
    f"candidate_seed{seed}_ema_primary_cases.json"
    for seed in SEEDS
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rerun-dir",
        type=Path,
        default=ROOT / "outputs/chip_hierarchy/v10_three_seed/reproducible_v1",
    )
    parser.add_argument(
        "--rerun-template",
        default="candidate_seed{seed}_ema_primary_repro_v44_lp2s.json",
    )
    parser.add_argument("--tolerance", type=float, default=1.0e-10)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_canonical_anchor_audit_v44.json",
    )
    return parser.parse_args()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def tree_digest(path: Path) -> str:
    entries = [
        f"{child.relative_to(path).as_posix()}:{digest(child)}"
        for child in sorted(path.iterdir())
        if child.is_file()
    ]
    return hashlib.sha256("\n".join(entries).encode("utf-8")).hexdigest()


def result_key(report: dict) -> str:
    keys = sorted(report["results"])
    if len(keys) != 1:
        raise ValueError(f"expected one result setting, found {keys}")
    return keys[0]


def metrics(report: dict) -> dict[str, float]:
    row = report["results"][result_key(report)]
    return {metric: float(row[metric]) for metric in METRICS}


def case_metrics(path: Path) -> dict[str, float]:
    report = read_json(path)
    rows = report[result_key({"results": report})]
    return {
        metric: float(np.mean([float(row[metric]) for row in rows]))
        for metric in METRICS
    }


def checkpoint_config(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "config": json.loads(json.dumps(payload.get("config"), default=str)),
        "step": payload.get("step"),
        "seed": payload.get("seed"),
        "has_ema": "ema" in payload,
    }


def close(first: float, second: float, tolerance: float) -> bool:
    return math.isclose(first, second, rel_tol=tolerance, abs_tol=tolerance)


def mean(values: list[dict[str, float]]) -> dict[str, float]:
    return {
        metric: float(np.mean([value[metric] for value in values]))
        for metric in METRICS
    }


def main() -> None:
    args = parse_args()
    aggregate = read_json(
        ROOT / "outputs/reports/chip_typed_hierarchy_three_seed_v3_ema.json"
    )
    trinity = read_json(
        ROOT / "outputs/reports/chip_trinity_d192_component_gap_v1.json"
    )["main_comparison"]["trinity"]
    current_rows = {}
    historical_rows = {}
    reruns = {}
    for seed in SEEDS:
        current_path = args.rerun_dir / args.rerun_template.format(seed=seed)
        old_cases = ROOT / OLD_CASES[seed]
        historical_rows[seed] = case_metrics(old_cases)
        if not current_path.is_file():
            reruns[str(seed)] = {
                "status": "missing",
                "path": str(current_path.relative_to(ROOT)),
            }
            continue
        report = read_json(current_path)
        current = metrics(report)
        current_rows[seed] = current
        eligible = (
            Path(report["checkpoint"]).resolve() == (ROOT / CHECKPOINTS[seed]).resolve()
            and Path(report["cache"]).resolve()
            == (ROOT / "outputs/cache/chip_official_test_100_v1").resolve()
            and bool(report.get("use_ema"))
            and report.get("projector") is True
            and int(report.get("projector_iterations", -1)) == 8
            and report.get("deterministic") is True
            and int(report.get("legalizer_time_limit_ms", -1)) == 2000
            and list(report.get("nfe", [])) == [4]
            and list(report.get("refiner_steps", [])) == [400]
        )
        historical_match = all(
            close(current[metric], historical_rows[seed][metric], args.tolerance)
            for metric in METRICS
        )
        reruns[str(seed)] = {
            "status": "complete",
            "path": str(current_path.relative_to(ROOT)),
            "eligible": eligible,
            "current": current,
            "historical": historical_rows[seed],
            "relative_difference_from_historical": {
                metric: current[metric] / historical_rows[seed][metric] - 1.0
                for metric in METRICS
            },
            "historical_match": historical_match,
            "settings": {
                "use_ema": bool(report.get("use_ema")),
                "projector": report.get("projector"),
                "projector_iterations": report.get("projector_iterations"),
                "deterministic": report.get("deterministic"),
                "legalizer_time_limit_ms": report.get("legalizer_time_limit_ms"),
                "nfe": report.get("nfe"),
                "refiner_steps": report.get("refiner_steps"),
                "refiner": report.get("refiner"),
            },
        }
    complete = len(current_rows) == len(SEEDS) and all(
        value.get("eligible") for value in reruns.values()
    )
    current_mean = mean([current_rows[seed] for seed in SEEDS]) if complete else None
    historical_mean = mean([historical_rows[seed] for seed in SEEDS])
    historical_matches = complete and all(
        value.get("historical_match") for value in reruns.values()
    )
    trinity_metrics = {
        "cost": float(trinity["mean_cost"]),
        "v_rel": float(trinity["mean_v_rel"]),
    }
    report = {
        "status": (
            "pass_historical_match"
            if complete and historical_matches
            else "pass_current_protocol_rebuild"
            if complete
            else "pending_three_seed_replay"
        ),
        "canonical_model": "typed_hierarchy_three_seed_ema",
        "superseded_model": (
            "The 1.2627975/0.0511892 graph-token values use a different checkpoint path."
        ),
        "primary_endpoint": (
            "NFE4 + 400-step linear refiner + projector_iterations=8 + "
            "scale_pack LP time limit 2000 ms"
        ),
        "historical_candidate_seed_mean": historical_mean,
        "aggregate_report_seed_mean": {
            metric: float(aggregate["seed_mean"]["candidate"][metric])
            for metric in METRICS
        },
        "current_candidate_seed_mean": current_mean,
        "trinity_reproducible": trinity_metrics,
        "current_gap_to_trinity": (
            {
                "cost_relative": current_mean["legalized_hard_cost"]
                / trinity_metrics["cost"]
                - 1.0,
                "v_rel_relative": current_mean["legalized_v_rel"]
                / trinity_metrics["v_rel"]
                - 1.0,
                "beats_cost": current_mean["legalized_hard_cost"]
                < trinity_metrics["cost"],
                "beats_v_rel": current_mean["legalized_v_rel"]
                < trinity_metrics["v_rel"],
            }
            if current_mean is not None
            else None
        ),
        "reruns": reruns,
        "artifacts": {
            "checkpoints": {
                str(seed): {
                    "path": CHECKPOINTS[seed],
                    "sha256": digest(ROOT / CHECKPOINTS[seed]),
                    **checkpoint_config(ROOT / CHECKPOINTS[seed]),
                }
                for seed in SEEDS
            },
            "cache": {
                "path": "outputs/cache/chip_official_test_100_v1",
                "tree_sha256": tree_digest(
                    ROOT / "outputs/cache/chip_official_test_100_v1"
                ),
            },
            "source_sha256": {
                path: digest(ROOT / path)
                for path in (
                    "scripts/evaluate_chip_refiner.py",
                    "scripts/evaluate_chip_official.py",
                    "src/floorflow/chip_flow/legalization.py",
                    "src/floorflow/chip_flow/hierarchical_flow.py",
                    "src/floorflow/chip_flow/hierarchical_flow_runtime.py",
                )
            },
        },
    }
    if complete:
        report["decision"] = (
            "freeze_typed_hierarchy_three_seed_ema_current_protocol"
            if historical_matches
            else "freeze_current_protocol_and_retain_historical_drift"
        )
    else:
        report["decision"] = "finish_current_protocol_three_seed_replay"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "decision": report["decision"],
                "current_candidate_seed_mean": current_mean,
                "current_gap_to_trinity": report["current_gap_to_trinity"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
