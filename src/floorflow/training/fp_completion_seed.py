"""Command definitions for the matched completion multi-seed sweep."""

from __future__ import annotations

import sys
from pathlib import Path


def build_stage_command(
    *,
    stage: str,
    seed: int,
    device: str,
    output_root: Path,
    data_root: Path,
    cache_root: Path,
    geometry_root: Path,
    batch_size: int,
    workers: int,
    limit_val: int,
) -> tuple[Path, list[str]]:
    output_dir = output_root / stage
    common = [
        "--data-root",
        str(data_root),
        "--output-dir",
        str(output_dir),
        "--batch-size",
        str(batch_size),
        "--workers",
        str(workers),
        "--limit-val",
        str(limit_val),
        "--device",
        device,
        "--seed",
        str(seed),
    ]
    if stage == "layout":
        return output_dir, [
            sys.executable,
            "scripts/train_layout_refiner.py",
            *common,
            "--cache-root",
            str(cache_root),
            "--layout-size",
            "224",
            "--epochs",
            "5",
        ]
    if stage == "structure":
        return output_dir, [
            sys.executable,
            "scripts/train_coarse_structure_refiner.py",
            *common,
            "--cache-root",
            str(cache_root),
            "--control-size",
            "128",
            "--epochs",
            "16",
            "--base-channels",
            "32",
            "--validate-every",
            "2",
        ]
    if stage == "furniture":
        return output_dir, [
            sys.executable,
            "scripts/train_coarse_furniture_refiner.py",
            *common,
            "--cache-root",
            str(cache_root),
            "--control-size",
            "192",
            "--epochs",
            "12",
            "--base-channels",
            "32",
            "--validate-every",
            "2",
        ]
    if stage == "support":
        return output_dir, [
            sys.executable,
            "scripts/train_coarse_support_refiner.py",
            *common,
            "--geometry-root",
            str(geometry_root),
            "--control-size",
            "192",
            "--output-size",
            "256",
            "--epochs",
            "12",
            "--base-channels",
            "32",
            "--validate-every",
            "2",
        ]
    raise ValueError(f"unknown stage: {stage}")


__all__ = ["build_stage_command"]
