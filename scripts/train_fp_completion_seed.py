#!/usr/bin/env python3
"""Run the matched completion-refiner training stack for one seed."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from floorflow.training.fp_completion_seed import build_stage_command

STAGE_ORDER = ("layout", "structure", "furniture", "support")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path(".data/cubicasa5k"))
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("outputs/cache/cubicasa5k_full_v2"),
    )
    parser.add_argument(
        "--geometry-root",
        type=Path,
        default=Path("outputs/cache/cubicasa5k_geometry_v1"),
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit-val", type=int, default=191)
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("layout", "structure", "furniture", "support"),
        default=("layout", "structure", "furniture", "support"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_root.exists() and not args.overwrite:
        raise FileExistsError(
            f"{args.output_root} exists; pass --overwrite to replace outputs"
        )
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "manifest.json"
    existing_records = {}
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing_records = {
            record["stage"]: record for record in existing.get("stages", [])
        }
    manifest = {
        "seed": args.seed,
        "device": args.device,
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "stages": [],
    }
    stage_records = dict(existing_records)
    for stage in args.stages:
        output_dir, command = build_stage_command(
            stage=stage,
            seed=args.seed,
            device=args.device,
            output_root=args.output_root,
            data_root=args.data_root,
            cache_root=args.cache_root,
            geometry_root=args.geometry_root,
            batch_size=args.batch_size,
            workers=args.workers,
            limit_val=args.limit_val,
        )
        if (output_dir / "best.pt").is_file() and not args.overwrite:
            stage_records[stage] = {
                "stage": stage,
                "status": "skipped_existing",
                "checkpoint": str(output_dir / "best.pt"),
            }
            manifest["stages"] = [
                stage_records[name]
                for name in STAGE_ORDER
                if name in stage_records
            ]
            continue
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / "train.log"
        started = time.perf_counter()
        record = {
            "stage": stage,
            "status": "running",
            "command": command,
            "log": str(log_path),
        }
        stage_records[stage] = record
        manifest["stages"] = [
            stage_records[name]
            for name in STAGE_ORDER
            if name in stage_records
        ]
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        with log_path.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(
                command,
                cwd=Path(__file__).resolve().parents[1],
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        record["seconds"] = time.perf_counter() - started
        record["returncode"] = completed.returncode
        checkpoint = output_dir / "best.pt"
        record["checkpoint"] = str(checkpoint)
        record["status"] = (
            "complete"
            if completed.returncode == 0 and checkpoint.is_file()
            else "failed"
        )
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        if record["status"] == "failed":
            manifest["status"] = "failed"
            manifest_path.write_text(
                json.dumps(manifest, indent=2) + "\n",
                encoding="utf-8",
            )
            raise SystemExit(completed.returncode or 1)
    all_complete = all(
        stage_records.get(stage, {}).get("status")
        in {"complete", "skipped_existing"}
        for stage in STAGE_ORDER
    )
    manifest["status"] = "complete" if all_complete else "partial"
    manifest["completed_at"] = datetime.now(UTC).isoformat()
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
