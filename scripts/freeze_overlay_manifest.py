#!/usr/bin/env python3
"""Freeze the selected deterministic overlay branch for reproducibility."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gate",
        type=Path,
        default=Path("outputs/reports/style_overlay_publication_gate_v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reports/style_overlay_manifest_v1.json"),
    )
    args = parser.parse_args()
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    paths = {
        "cubicasa_checkpoint": Path(
            "outputs/train/distribution_style_prompt_r94_v2/best.pt"
        ),
        "cubicasa_style_bank": Path(
            "outputs/style_banks/cubicasa_train_v1.json"
        ),
        "retrieval_weights": Path(
            "outputs/reports/a1_retrieval_weights_calib191.json"
        ),
        "taxonomy": Path("outputs/cache/cubicasa5k_full_v2/taxonomy.json"),
        "train_split": Path(".data/cubicasa5k/train.txt"),
        "val_split": Path(".data/cubicasa5k/val.txt"),
        "test_split": Path(".data/cubicasa5k/test.txt"),
        "gate": args.gate,
        "overlay_evaluator": Path("scripts/evaluate_distribution_overlay.py"),
        "overlay_composer": Path("src/floorflow/inference/distribution_overlay.py"),
        "retrieval_prior": Path("src/floorflow/inference/retrieval_prior.py"),
        "style_separation": Path(
            "outputs/reports/distribution_r94_overlay_style_separation_val32_w05.json"
        ),
        "val_all_styles": Path(
            "outputs/reports/distribution_r94_overlay_all_styles_holdout300.json"
        ),
        "test_all_styles": Path(
            "outputs/reports/distribution_r94_overlay_all_styles_test503_w05.json"
        ),
    }
    report = {
        "status": "frozen",
        "created_utc": datetime.now(UTC).isoformat(),
        "decoder": "pixel_element_retrieval_overlay",
        "retrieval_prior_weight": 0.5,
        "retrieval_top_k": 1,
        "single_sample": True,
        "best_of_n_used": False,
        "target_derived_control": False,
        "style_source": "train-only bank selected explicitly at inference",
        "gate_status": gate["status"],
        "artifacts": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in paths.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
