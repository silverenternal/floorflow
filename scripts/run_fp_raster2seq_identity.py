#!/usr/bin/env python3
"""Run Raster2Seq reconstruction once on each unique CubiCasa test503 input."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.edit_detection import decode_visible_controls
from floorflow.evaluation.raster2seq_external import (
    decode_raster2seq_sequence,
    render_raster2seq_input,
    render_raster2seq_official_style,
    render_raster2seq_predictions,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_fp_raster2seq_matched_edits import load_model, to_tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT
        / "outputs/external_baselines/raster2seq/cubicasa5k/checkpoint.pth",
    )
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=ROOT / "outputs/cache/cubicasa5k_full_v2/taxonomy.json",
    )
    parser.add_argument(
        "--raster-root",
        type=Path,
        default=ROOT / "outputs/eval/fp_hard_metrics_full_test503_seed19_v1/rasters",
    )
    parser.add_argument(
        "--control-root",
        type=Path,
        default=ROOT / "outputs/control_contracts/oracle_v1",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/external_baselines/raster2seq/identity_test503_v1",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--renderer",
        choices=("official_style", "neutral"),
        default="official_style",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def git_output(repo: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def case_paths(root: Path) -> list[Path]:
    paths = sorted(root.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"no raster inputs under {root}")
    return paths


def control_paths(root: Path, split: str) -> dict[str, Path]:
    manifest_path = root / split / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        str(row["name"]).replace("/", "__"): Path(row["file"])
        for row in payload["rows"]
    }


def provenance(checkpoint: Path) -> dict:
    repo = ROOT / "third_party/Raster2Seq"
    license_path = repo / "LICENSE"
    dirty = git_output(repo, "status", "--short", "--untracked-files=no")
    return {
        "repository": git_output(repo, "remote", "get-url", "origin"),
        "local_commit": git_output(repo, "rev-parse", "HEAD"),
        "remote_head_at_run": git_output(repo, "ls-remote", "origin", "HEAD"),
        "dirty_tracked_files": dirty.splitlines() if dirty else [],
        "license": (
            license_path.read_text(encoding="utf-8").splitlines()[0]
            if license_path.is_file()
            else None
        ),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    outputs_root = args.output_dir / "predictions"
    outputs_root.mkdir(parents=True, exist_ok=True)
    taxonomy = FloorplanTaxonomy.read(args.taxonomy)
    cases = case_paths(args.raster_root)
    controls = control_paths(args.control_root, args.split)
    if args.limit is not None:
        cases = cases[: args.limit]
    device = torch.device(args.device)
    model, image_size = load_model(args.checkpoint, device)
    rows = []
    for start in range(0, len(cases), args.batch_size):
        batch_paths = cases[start : start + args.batch_size]
        images = []
        payloads = []
        for path in batch_paths:
            with np.load(path) as payload:
                target = payload["target"].astype(np.uint8, copy=False)
                footprint = payload["footprint"].astype(np.uint8, copy=False)
            payloads.append((target, footprint))
            if path.stem not in controls:
                raise KeyError(f"missing before-edit control for {path.stem}")
            before_labels = decode_visible_controls(
                controls[path.stem],
                footprint.astype(bool),
                taxonomy,
            )
            renderer = (
                render_raster2seq_official_style
                if args.renderer == "official_style"
                else render_raster2seq_input
            )
            images.append(renderer(before_labels, taxonomy))
        inputs = to_tensor(images, device)
        predictions = model.forward_inference(inputs, use_cache=True)
        for index, path in enumerate(batch_paths):
            target, footprint = payloads[index]
            output_path = outputs_root / path.name
            if not (args.skip_existing and output_path.is_file()):
                polygons, labels = decode_raster2seq_sequence(
                    predictions["gen_out"][index],
                    predictions["pred_room_logits"][index],
                    taxonomy,
                    image_size=image_size,
                )
                rendered = render_raster2seq_predictions(
                    polygons,
                    labels,
                    target.shape,
                    taxonomy,
                )
                np.savez_compressed(
                    output_path,
                    prediction=rendered,
                    footprint=footprint,
                    target=target,
                )
            with np.load(output_path) as result:
                prediction = result["prediction"].astype(np.uint8, copy=False)
            rows.append(
                {
                    "index": start + index,
                    "name": path.stem.replace("__", "/").replace("_Floor-", "__Floor-"),
                    "control": str(controls[path.stem]),
                    "control_sha256": sha256(controls[path.stem]),
                    "input": str(path),
                    "prediction": str(output_path),
                    "input_sha256": sha256(path),
                    "prediction_sha256": sha256(output_path),
                    "prediction_nonzero_pixels": int(np.count_nonzero(prediction)),
                }
            )
        print(f"reconstructed {min(start + len(batch_paths), len(cases))}/{len(cases)}", flush=True)
    report = {
        "status": "complete",
        "scope": f"identity reconstruction on CubiCasa {args.split} rasters",
        "split": args.split,
        "cases": len(rows),
        "batch_size": args.batch_size,
        "device": str(device),
        "image_size": image_size,
        "renderer": args.renderer,
        "deterministic_single_sample": True,
        "best_of_n_used": False,
        "provenance": provenance(args.checkpoint),
        "rows": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "complete", "cases": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
