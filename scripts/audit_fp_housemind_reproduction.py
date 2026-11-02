#!/usr/bin/env python3
"""Record HouseMind asset availability and the current reproduction blocker."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("/tmp/housemind_data_test.zip"),
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("/tmp/housemind_weights.zip"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reports/fp_housemind_reproduction_status_v1.json"),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def zip_manifest(path: Path) -> dict:
    if not path.is_file():
        return {"path": str(path), "available": False}
    with zipfile.ZipFile(path) as archive:
        files = [name for name in archive.namelist() if not name.endswith("/")]
    return {
        "path": str(path),
        "available": True,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "files": len(files),
        "preview": files[:12],
    }


def main() -> None:
    args = parse_args()
    benchmark = zip_manifest(args.benchmark)
    weights = zip_manifest(args.weights)
    benchmark_ready = (
        benchmark.get("available", False)
        and benchmark.get("files", 0) >= 4 * 2308
    )
    weights_ready = (
        weights.get("available", False)
        and any("model.safetensors" in name for name in weights["preview"])
    )
    report = {
        "status": (
            "assets_ready_official_code_unreleased"
            if benchmark_ready and weights_ready
            else "assets_incomplete_official_code_unreleased"
        ),
        "method": "HouseMind",
        "paper": "https://arxiv.org/abs/2603.11640",
        "project_page": "https://housemind.github.io/",
        "benchmark_source": (
            "https://cloud.tsinghua.edu.cn/f/7cbe9e7039f843e78236/"
        ),
        "weights_source": (
            "https://cloud.tsinghua.edu.cn/f/e1e91fedfd3f4d80b8f0/"
        ),
        "official_code_status": "Code Soon on the official project page",
        "benchmark": benchmark,
        "weights": weights,
        "released_assets": {
            "joint_test_cases": 2308,
            "layers": [
                "outline_256",
                "before_edit_256",
                "after_edit_256",
                "gt_256",
            ],
            "weights": [
                "housemind-omni",
                "outline_vqvae",
                "room_vqvae",
            ],
        },
        "reproduction_blocker": (
            "The official inference code and prompt/token assembly are not "
            "released. Weights and data alone are insufficient to reconstruct "
            "the authoritative HouseMind-E edit pipeline without guessing its "
            "serialization contract."
        ),
        "decision": (
            "Do not claim a HouseMind same-benchmark result. Preserve the "
            "CubiCasa protocol anchor and matched Raster2Seq comparison."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
