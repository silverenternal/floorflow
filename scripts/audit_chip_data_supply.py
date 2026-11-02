"""CPU-only manifest freeze and data-supply audit for chip training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from floorflow.chip_flow.cache import PackedCaseStore
from floorflow.chip_flow.data_manifest import (
    build_data_manifest,
    cache_fingerprint,
    sha256_file,
    write_data_manifest,
)
from floorflow.chip_flow.eval_curve_manifest import (
    build_eval_curve_manifest,
    write_eval_curve_manifest,
)
from floorflow.chip_flow.training_runtime import write_json_atomic


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("outputs/cache/chip_floorset_50k_train_12k_dev_v2"),
    )
    parser.add_argument(
        "--data-manifest",
        type=Path,
        default=Path("configs/chip_data_manifest_50k_v1.json"),
    )
    parser.add_argument(
        "--curve-manifest",
        type=Path,
        default=Path("configs/chip_eval_curve_v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reports/chip_data_supply_audit_v1.json"),
    )
    parser.add_argument("--curve-per-count", type=int, default=5)
    parser.add_argument("--eval-curve-seed", type=int, default=0)
    parser.add_argument(
        "--range-cache",
        type=Path,
        default=Path(
            "outputs/cache/chip_floorset_200k_train_12k_dev_range_v1/train"
        ),
    )
    parser.add_argument(
        "--range-config",
        type=Path,
        default=Path("configs/chip_train_scaling_range20k_v1.json"),
    )
    parser.add_argument(
        "--legacy-config",
        type=Path,
        default=Path("configs/chip_train_scaling_20k_v1.json"),
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    train = PackedCaseStore(args.cache / "train")
    dev = PackedCaseStore(args.cache / "dev")
    manifest = build_data_manifest(args.cache / "train", args.cache / "dev")
    manifest["range_train"] = cache_fingerprint(args.range_cache)
    manifest["range_config_sha256"] = sha256_file(args.range_config)
    write_data_manifest(args.data_manifest, manifest)
    curve = build_eval_curve_manifest(
        dev,
        per_count=args.curve_per_count,
        eval_seed=args.eval_curve_seed,
    )
    write_eval_curve_manifest(args.curve_manifest, curve)
    dev_counts = dev.arrays["n"]
    range_store = PackedCaseStore(args.range_cache)
    range_config = json.loads(args.range_config.read_text(encoding="utf-8"))
    legacy_config = json.loads(args.legacy_config.read_text(encoding="utf-8"))
    range_indices = range_config["indices"]
    range_counts = range_store.arrays["n"][range_indices]
    train_max = train.arrays["n"].max()
    report = {
        "version": "chip-data-supply-audit-v1",
        "status": "pass",
        "device": "cpu",
        "train": {
            "cases": len(train),
            "min_blocks": int(train.arrays["n"].min()),
            "max_blocks": int(train_max),
            "count_groups": len(train.by_block_count),
        },
        "dev": {
            "cases": len(dev),
            "min_blocks": int(dev_counts.min()),
            "max_blocks": int(dev_counts.max()),
            "count_groups": len(dev.by_block_count),
            "cases_over_train_max": int((dev_counts > train_max).sum()),
            "fraction_over_train_max": float((dev_counts > train_max).mean()),
        },
        "shared_eval_curve": {
            "cases": len(curve["indices"]),
            "eval_seed": args.eval_curve_seed,
            "per_count": args.curve_per_count,
        },
        "range_supply": {
            "cases": len(range_store),
            "min_blocks": int(range_store.arrays["n"].min()),
            "max_blocks": int(range_store.arrays["n"].max()),
            "count_groups": len(range_store.by_block_count),
            "subset_cases": len(range_indices),
            "subset_cases_over_60": int((range_counts > 60).sum()),
            "subset_max_blocks": int(range_counts.max()),
            "matched_exposure_with_legacy": len(range_indices)
            == len(legacy_config["indices"]),
            "config_sha256": sha256_file(args.range_config),
        },
        "manifests": {
            "data": str(args.data_manifest),
            "curve": str(args.curve_manifest),
        },
        "boundary": "CPU supply audit only; no training or quality claim.",
    }
    write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
