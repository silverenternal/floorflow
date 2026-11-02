#!/usr/bin/env python3
"""Inventory control-gated inputs, rendered rows, and scorer outputs."""

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
        default=Path(
            "outputs/reports/fp_control_gated_reproducibility_manifest_v1.json"
        ),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(root: Path, pattern: str = "*") -> dict:
    files = sorted(path for path in root.rglob(pattern) if path.is_file())
    return {
        "path": str(root),
        "files": len(files),
        "bytes": sum(path.stat().st_size for path in files),
        "entries": {
            str(path): {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in files
        },
    }


def main() -> None:
    args = parse_args()
    rendered = {
        f"seed{seed}": inventory(
            Path(f"outputs/eval/fp_control_gated_local_test503_seed{seed}_v1")
        )
        for seed in (19, 20, 21, 22, 23)
    }
    external = inventory(
        Path("outputs/external_baselines/raster2seq/test503_matched_v1")
    )
    scorer = {
        f"seed{seed}": inventory(
            Path("outputs/eval"),
            pattern=f"fp_control_gated_topology_fpeval_*_seed{seed}_v1.json",
        )
        for seed in (19, 20, 21, 22, 23)
    }
    seed_configs = {
        f"seed{seed}": inventory(
            Path(f"outputs/train/fp_completion_seed{seed}")
        )
        for seed in (20, 21, 22, 23)
    }
    report = {
        "status": "complete",
        "control_gated_output_contract": (
            "The candidate operates on existing pre/post raster predictions "
            "and the declared control-change mask. It does not introduce new "
            "network logits."
        ),
        "raw_logits_policy": (
            "Raw logits are not applicable to the control-gated compositor. "
            "The package includes its complete rendered inputs/outputs and the "
            "upstream checkpoint paths needed to regenerate them."
        ),
        "rendered_rows": rendered,
        "external_baseline": external,
        "scorer_reports": scorer,
        "seed_configs": seed_configs,
        "seed19_configuration_note": (
            "Seed19 predates the fp_completion_seed20-23 manifest wrapper; its "
            "metric and checkpoint provenance remains in the hard-metric report."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "rendered_files": sum(
                    item["files"] for item in rendered.values()
                ),
                "external_files": external["files"],
                "scorer_files": sum(item["files"] for item in scorer.values()),
                "seed_config_files": sum(
                    item["files"] for item in seed_configs.values()
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
