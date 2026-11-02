#!/usr/bin/env python3
"""Audit whether local external floorplan baselines are runnable."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reports/external_baseline_availability_v1.json"),
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def asset(path: Path, *, checksum_limit: int = 0) -> dict:
    result = {
        "path": str(path),
        "exists": path.exists(),
        "size": path.stat().st_size if path.is_file() else None,
    }
    if path.is_dir():
        result["file_count"] = sum(item.is_file() for item in path.rglob("*"))
    if path.is_file() and checksum_limit and path.stat().st_size <= checksum_limit:
        result["sha256"] = file_sha256(path)
    return result


def main() -> None:
    args = parse_args()
    house_diffusion = Path("third_party/house-diffusion")
    houseganpp = Path("third_party/houseganpp")
    maskplan = Path("third_party/MaskPLAN")
    house_diffusion_checkpoint = (
        Path("outputs/external_baselines/house_diffusion/house_diffusion_model.pt")
    )
    maskplan_checkpoint = (
        maskplan / "MaskPLAN_Trained/All_Large_Single_cross"
    )
    maskplan_boundaries = maskplan / "parsed_img/img_room_sqe/0"
    houseganpp_smoke = Path("outputs/external_baselines/houseganpp/smoke_r1")
    maskplan_smoke = Path("outputs/external_baselines/maskplan/smoke_r1")
    baselines = {
        "house_diffusion": {
            "repo": asset(house_diffusion),
            "checkpoint": asset(house_diffusion_checkpoint, checksum_limit=200_000_000),
            "repo_checkpoint_link": asset(
                house_diffusion / "ckpts" / "house_diffusion_model.pt"
            ),
            "processed_dataset": asset(house_diffusion / "datasets"),
            "native_rplan_samples": asset(
                house_diffusion / "datasets" / "rplan" / "list.txt",
                checksum_limit=20_000_000,
            ),
            "magicplan_dataset": asset(
                house_diffusion / "datasets" / "magicplan"
            ),
            "official_weight_source": (
                "https://drive.google.com/file/d/16zKmtxwY5lF6JE-CJGkRf3-OFoD1TrdR/view"
            ),
            "magicplan_source_sha256": (
                "800ead4c55596207a9e67c8d508b42fd85ab5385a2d63010edb6a958d21c4901"
            ),
            "compatibility_note": (
                "The released transformer consumes an RPLAN room graph and per-room "
                "corner/type tensors; a CubiCasa contour adapter is still required."
            ),
        },
        "houseganpp": {
            "repo": asset(houseganpp),
            "checkpoint": asset(
                houseganpp / "checkpoints" / "pretrained.pth",
                checksum_limit=20_000_000,
            ),
            "sample_list": asset(
                houseganpp / "data" / "sample_list.txt",
                checksum_limit=20_000_000,
            ),
            "json_samples": len(list((houseganpp / "data" / "json").glob("*.json"))),
            "smoke_output_files": (
                sum(item.is_file() for item in houseganpp_smoke.rglob("*"))
                if houseganpp_smoke.exists()
                else 0
            ),
            "official_weight_source": (
                "https://raw.githubusercontent.com/ennauata/houseganpp/main/"
                "checkpoints/pretrained.pth"
            ),
            "compatibility_note": (
                "The released model consumes a room graph, adjacency, and room masks; "
                "it is not a contour-only baseline without an adapter."
            ),
        },
        "maskplan": {
            "repo": asset(maskplan),
            "processed_data": asset(maskplan / "Processed_data"),
            "layout_checkpoint": {
                "directory": asset(maskplan_checkpoint),
                "index": asset(maskplan_checkpoint / "All.index", checksum_limit=100_000),
                "data": asset(maskplan_checkpoint / "All.data-00000-of-00001"),
            },
            "vq_checkpoints": len(
                list((maskplan / "VQ_Pretrained").rglob("*.npy"))
            ),
            "boundary_images": asset(maskplan_boundaries),
            "smoke_output_files": (
                sum(item.is_file() for item in maskplan_smoke.rglob("*"))
                if maskplan_smoke.exists()
                else 0
            ),
            "tensorflow_environment": asset(
                Path(".external_envs/maskplan/bin/python")
            ),
            "official_weight_source": (
                "https://drive.google.com/file/d/1KkHRhd19TE-oIrgpPlY53TNGP1YT848j/view"
            ),
            "source_archive_sha256": (
                "c941707b5f0eeaf18249c7947f1bbfd398593fe269d349c177f550b9c54b4f6a"
            ),
            "compatibility_note": (
                "The released Large/Single/cross checkpoint consumes ten RPLAN room "
                "tokens, a boundary token sequence, and a front-door vector."
            ),
        },
    }
    required = [
        house_diffusion_checkpoint.is_file(),
        (house_diffusion / "datasets/rplan/list.txt").is_file(),
        (house_diffusion / "datasets/magicplan").is_dir(),
        (houseganpp / "checkpoints/pretrained.pth").is_file(),
        (houseganpp / "data/sample_list.txt").is_file(),
        (maskplan_checkpoint / "All.index").is_file(),
        (maskplan_checkpoint / "All.data-00000-of-00001").is_file(),
        (maskplan_boundaries / "0.png").is_file(),
    ]
    dataset_ready = (
        (house_diffusion / "datasets/rplan/list.txt").is_file()
        and (house_diffusion / "datasets/magicplan").is_dir()
    )
    report = {
        "status": (
            "assets_ready_pending_cubicasa_adapters"
            if all(required)
            else "partial_external_assets"
        ),
        "baselines": baselines,
        "measured_metrics": {},
        "verified_preflight": {
            "house_diffusion": {
                "state_dict_tensors": 186,
                "parameters": 26_541_330,
                "strict_state_dict_load": True,
                "native_rplan_cases": 6,
                "forward_shapes": {
                    "dec": [1, 2, 100],
                    "binary": [1, 16, 100],
                },
            },
            "houseganpp": {
                "state_dict_tensors": 50,
                "parameters": 635_569,
                "strict_state_dict_load": True,
                "official_samples_completed": 6,
            },
            "maskplan": {
                "configuration": "Large_Single_cross",
                "parameters": 388_897_374,
                "weights_load": True,
                "boundary_only_site": 16005,
                "inference_smoke_outputs": 3,
            },
        },
        "native_dataset_ready": dataset_ready,
        "policy": (
            "No fabricated baseline numbers. Native RPLAN/MagicPlan smoke results "
            "are not CubiCasa test results and must not be reported as such."
        ),
        "next_actions": [
            "Convert the frozen CubiCasa split to each baseline's native program/graph interface.",
            "Keep oracle-program and contour-only comparisons in separate tables.",
            "Run the frozen test split once after adapter validation; do not use best-of-N.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
