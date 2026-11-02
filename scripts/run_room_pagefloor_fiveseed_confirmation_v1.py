#!/usr/bin/env python3
"""Run the promoted room mechanism on full train and a disjoint five-seed holdout."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "outputs/reports"


def _run(command: list[str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def _complete_report(path: Path) -> bool:
    if not path.is_file():
        return False
    return json.loads(path.read_text(encoding="utf-8")).get("status") == "complete"


def _assert_gpu_available(device: str) -> None:
    if not device.startswith("cuda:"):
        return
    index = int(device.split(":", 1)[1])
    gpu_rows = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).splitlines()
    uuids = {
        str(row.split(",", 1)[1]).strip()
        for row in gpu_rows
        if int(str(row.split(",", 1)[0]).strip()) == index
    }
    app_rows = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).splitlines()
    active = [
        row.strip()
        for row in app_rows
        if str(row.split(",", 1)[0]).strip() in uuids
    ]
    if active:
        raise RuntimeError(
            f"{device} has active compute processes; refusing to interfere: "
            + "; ".join(active)
        )


def _train_command(args, seed: int) -> tuple[Path, Path]:
    output_dir = args.output_root / f"seed{seed}"
    command = [
        sys.executable,
        "scripts/train_raster_instance_control.py",
        "--output-dir",
        str(output_dir),
        "--full-train",
        "--limit-val",
        "384",
        "--val-offset",
        "0",
        "--train-sampling",
        "seeded",
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--workers",
        str(args.workers),
        "--page-floor-crop",
        "--query-set-head",
        "--query-count-gate",
        "--query-set-weight",
        "0.5",
        "--existence-loss-weight",
        "0.5",
        "--selection-mode",
        "dense_relative_identity",
        "--selection-count-weight",
        "0.005",
        "--train-encoder",
        "--initialize-from",
        str(args.initialize_from),
        "--preserve-room-encoder",
        "--learning-rate",
        "0.0002",
        "--encoder-learning-rate",
        "0.00005",
        "--precision",
        args.precision,
        "--channels-last",
        "--triton-kernels",
        "--seed",
        str(seed),
        "--device",
        args.device,
    ]
    if args.compile_model:
        command.extend(["--compile-model", "--compile-mode", "reduce-overhead"])
    return output_dir, command


def _evaluation_command(
    args,
    *,
    checkpoint: Path,
    offset: int,
    limit: int,
    output: Path,
    query_count_gate: bool,
) -> list[str]:
    command = [
        sys.executable,
        "scripts/evaluate_raster_instance_control.py",
        "--checkpoint",
        str(checkpoint),
        "--offset",
        str(offset),
        "--limit",
        str(limit),
        "--batch-size",
        str(args.batch_size),
        "--workers",
        str(args.workers),
        "--page-floor-crop",
        "--device",
        args.device,
        "--output",
        str(output),
    ]
    if query_count_gate:
        command.append("--query-count-gate")
    return command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[19, 23, 29, 31, 37])
    parser.add_argument(
        "--initialize-from",
        type=Path,
        default=(
            ROOT
            / "outputs/train/room_pagefloor_querygate_screen256_seed19_v1/best.pt"
        ),
    )
    parser.add_argument(
        "--parent-checkpoint",
        type=Path,
        default=(
            ROOT
            / "outputs/train/room_program_supervision_only_fast64_v2/best.pt"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs/train/room_pagefloor_confirmation_full5122_v1",
    )
    parser.add_argument(
        "--report-root",
        type=Path,
        default=REPORTS,
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--allow-concurrent-gpu",
        action="store_true",
        help="Bypass the default refusal to share the target GPU.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    for path in (args.initialize_from, args.parent_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.allow_concurrent_gpu:
        _assert_gpu_available(args.device)
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.report_root.mkdir(parents=True, exist_ok=True)
    parent_cal = args.report_root / "room_pagefloor_static_parent_cal384_cases_v1.json"
    parent_holdout = (
        args.report_root / "room_pagefloor_static_parent_holdout107_cases_v1.json"
    )
    parent_jobs = (
        (parent_cal, 0, 384),
        (parent_holdout, 384, 107),
    )
    for output, offset, limit in parent_jobs:
        if args.skip_existing and output.is_file():
            continue
        _run(
            _evaluation_command(
                args,
                checkpoint=args.parent_checkpoint,
                offset=offset,
                limit=limit,
                output=output,
                query_count_gate=False,
            )
        )

    rows = []
    for seed in args.seeds:
        output_dir, train_command = _train_command(args, seed)
        report_path = output_dir / "report.json"
        checkpoint = output_dir / "best.pt"
        if not (
            args.skip_existing
            and checkpoint.is_file()
            and _complete_report(report_path)
        ):
            _run(train_command)
        if not checkpoint.is_file() or not _complete_report(report_path):
            raise RuntimeError(f"seed {seed} did not produce a complete checkpoint")
        best_blend = float(
            json.loads(report_path.read_text(encoding="utf-8"))["best_blend"]
        )
        cal = args.report_root / f"room_pagefloor_confirmation_cal384_seed{seed}_v1.json"
        holdout = (
            args.report_root
            / f"room_pagefloor_confirmation_holdout107_seed{seed}_v1.json"
        )
        jobs = ((cal, 0, 384), (holdout, 384, 107))
        for output, offset, limit in jobs:
            if args.skip_existing and output.is_file():
                continue
            _run(
                _evaluation_command(
                    args,
                    checkpoint=checkpoint,
                    offset=offset,
                    limit=limit,
                    output=output,
                    query_count_gate=True,
                )
            )
        rows.append(
            {
                "seed": seed,
                "checkpoint": str(checkpoint),
                "report": str(report_path),
                "best_blend": best_blend,
                "calibration_cases": str(cal),
                "holdout_cases": str(holdout),
            }
        )
    manifest = {
        "schema_version": "room-pagefloor-fiveseed-confirmation-v1",
        "train_cases": 5122,
        "calibration": {"offset": 0, "cases": 384},
        "disjoint_holdout": {"offset": 384, "cases": 107},
        "canonical_warm_start": str(args.initialize_from),
        "parent_checkpoint": str(args.parent_checkpoint),
        "seeds": rows,
    }
    manifest_path = args.report_root / "room_pagefloor_confirmation_manifest_v1.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
