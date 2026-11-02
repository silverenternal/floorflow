"""Evaluate validation-locked eight-view Raster2Seq consensus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "third_party/Raster2Seq")]

from datasets import build_dataset
from evaluations.rplan_eval.Evaluator import Evaluator_RPlan

from scripts.build_raster2seq_tta_dataset_v1 import transform_polygon
from scripts.evaluate_raster2seq_identity_refiner_v1 import (
    _case_ground_truth,
    _summary,
)
from scripts.evaluate_raster2seq_tta_consensus_v1 import (
    RUNS as FOUR_VIEW_RUNS,
)
from scripts.evaluate_raster2seq_tta_consensus_v1 import (
    _case_metrics,
    _paired,
    agreement_score,
)

OPENING_IDS = frozenset({9, 10})
INVERSE = {
    "hflip": "hflip",
    "vflip": "vflip",
    "rot180": "rot180",
    "rot90": "rot270",
    "rot270": "rot90",
    "transpose": "transpose",
    "transverse": "transverse",
}
STRATEGIES = (
    "original",
    "hflip_only",
    "vflip_only",
    "rot180_only",
    "rot90_only",
    "rot270_only",
    "transpose_only",
    "transverse_only",
    "case_consistency_8",
    "count_consensus_8",
    "count_consensus_8_t1",
    "count_consensus_8_t2",
    "count_consensus_8_t3",
    "count_agreement_8",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=ROOT / "outputs/cache/raster2seq_cc5k_val_refined_v1",
    )
    parser.add_argument("--eval-set", choices=("val", "test"), default="val")
    parser.add_argument(
        "--run-prefix",
        default="val503",
    )
    parser.add_argument(
        "--suffix",
        default="alpha075_v1",
    )
    parser.add_argument("--force-strategy", choices=STRATEGIES)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/raster2seq_dihedral_consensus_val_v1.json",
    )
    return parser.parse_args()


def run_paths(prefix: str, suffix: str, eval_set: str) -> dict[str, tuple[Path, str | None]]:
    base = ROOT / "outputs/external_baselines/raster2seq"
    if eval_set == "val":
        return {
            "original": (
                base / "official_eval_val503_blend_0.75_v1/jsons",
                None,
            ),
            "hflip": (
                base / f"tta_{prefix}_hflip_{suffix}/jsons",
                "hflip",
            ),
            "vflip": (
                base / f"tta_{prefix}_vflip_{suffix}/jsons",
                "vflip",
            ),
            "rot180": (
                base / f"tta_{prefix}_rot180_{suffix}/jsons",
                "rot180",
            ),
            "rot90": (
                base / f"tta_{prefix}_rot90_{suffix}/jsons",
                "rot90",
            ),
            "rot270": (
                base / f"tta_{prefix}_rot270_{suffix}/jsons",
                "rot270",
            ),
            "transpose": (
                base / f"tta_{prefix}_transpose_{suffix}/jsons",
                "transpose",
            ),
            "transverse": (
                base / f"tta_{prefix}_transverse_{suffix}/jsons",
                "transverse",
            ),
        }
    original = base / "residential_identity_test509_alpha075_v1/jsons"
    if not original.is_dir():
        original = FOUR_VIEW_RUNS["original"][0]
    return {
        "original": (original, None),
        "hflip": (base / f"tta_{prefix}_hflip_{suffix}/jsons", "hflip"),
        "vflip": (base / f"tta_{prefix}_vflip_{suffix}/jsons", "vflip"),
        "rot180": (base / f"tta_{prefix}_rot180_{suffix}/jsons", "rot180"),
        "rot90": (base / f"tta_{prefix}_rot90_{suffix}/jsons", "rot90"),
        "rot270": (base / f"tta_{prefix}_rot270_{suffix}/jsons", "rot270"),
        "transpose": (
            base / f"tta_{prefix}_transpose_{suffix}/jsons",
            "transpose",
        ),
        "transverse": (
            base / f"tta_{prefix}_transverse_{suffix}/jsons",
            "transverse",
        ),
    }


def load_runs(
    paths: dict[str, tuple[Path, str | None]],
    image_id: int,
    width: int,
    height: int,
) -> dict[str, list[dict]]:
    output = {}
    for name, (directory, transform) in paths.items():
        items = json.loads(
            (directory / f"{image_id:05d}.json").read_text(encoding="utf-8")
        )
        output[name] = [
            {
                **item,
                "segmentation": transform_polygon(
                    np.asarray(item["segmentation"], dtype=float)
                    .reshape(-1)
                    .tolist(),
                    width,
                    height,
                    INVERSE[transform],
                )
                if transform is not None
                else np.asarray(item["segmentation"], dtype=float)
                .reshape(-1)
                .tolist(),
            }
            for item in items
        ]
    return output


def room_count(items: list[dict]) -> int:
    return sum(int(item["category_id"]) not in OPENING_IDS for item in items)


def select_consensus(items: dict[str, list[dict]], threshold: int) -> str:
    scores = {
        source: agreement_score(items, source)
        for source in items
    }
    counts = {source: room_count(rows) for source, rows in items.items()}
    median_count = float(np.median(list(counts.values())))
    selected = min(
        counts,
        key=lambda source: (
            abs(counts[source] - median_count),
            source != "original",
            -scores[source],
        ),
    )
    return (
        selected
        if abs(counts[selected] - median_count) <= threshold
        else "original"
    )


def fused_items(
    items: dict[str, list[dict]],
    strategy: str,
) -> list[dict]:
    if strategy == "original":
        return items["original"]
    if strategy.endswith("_only"):
        return items[strategy.removesuffix("_only")]
    scores = {source: agreement_score(items, source) for source in items}
    if strategy == "case_consistency_8":
        return items[max(scores, key=scores.get)]
    if strategy == "count_agreement_8":
        counts = {source: room_count(rows) for source, rows in items.items()}
        median_count = float(np.median(list(counts.values())))
        eligible = [
            source
            for source, count in counts.items()
            if abs(count - median_count) <= 2.0
        ]
        return items[max(eligible, key=lambda source: scores[source])]
    threshold = (
        int(strategy.rsplit("_t", 1)[1])
        if "_t" in strategy
        else 2
    )
    return items[select_consensus(items, threshold)]


def main() -> None:
    args = parse_args()
    paths = run_paths(args.run_prefix, args.suffix, args.eval_set)
    missing = [
        str(path.relative_to(ROOT))
        for path, _ in paths.values()
        if not path.is_dir()
    ]
    if missing:
        raise FileNotFoundError(f"missing inference directories: {missing}")
    dataset = build_dataset(
        args.eval_set,
        SimpleNamespace(
            dataset_name="cubicasa",
            dataset_root=str(args.dataset_root),
            image_size=256,
            image_norm=False,
            poly2seq=True,
            num_bins=32,
            seq_len=512,
            add_cls_token=False,
            per_token_sem_loss=True,
            mask_format="polygon",
            disable_image_transform=False,
            semantic_classes=12,
            wd_only=False,
        ),
    )
    evaluator = Evaluator_RPlan(
        disable_overlapping_filter=True,
        iou_thres=0.5,
        wd_as_line=True,
    )
    strategies = (
        tuple(dict.fromkeys(("original", args.force_strategy)))
        if args.force_strategy
        else STRATEGIES
    )
    rows = {strategy: [] for strategy in strategies}
    diagnostics = {}
    for index in range(len(dataset)):
        record = dataset[index]
        image_id = int(record["image_id"])
        runs = load_runs(paths, image_id, 256, 256)
        diagnostics[str(image_id)] = {
            "agreement": {
                source: agreement_score(runs, source) for source in runs
            },
            "room_counts": {
                source: room_count(items) for source, items in runs.items()
            },
        }
        ground_truth = _case_ground_truth(record)
        for strategy in strategies:
            rows[strategy].append(
                {
                    "image_id": image_id,
                    **_case_metrics(
                        fused_items(runs, strategy),
                        ground_truth,
                        evaluator,
                    ),
                }
            )
    summaries = {
        strategy: _summary(values) for strategy, values in rows.items()
    }
    original = summaries["original"]
    eligible = {
        strategy: values
        for strategy, values in summaries.items()
        if strategy != "original"
        and values["room_f1"] >= original["room_f1"]
        and values["room_iou"] >= original["room_iou"]
        and values["corner_f1"] >= original["corner_f1"]
    }
    selected = args.force_strategy or max(
        eligible or {"original": original},
        key=lambda name: (
            summaries[name]["room_f1"],
            summaries[name]["room_iou"],
        ),
    )
    comparisons = {}
    if selected in rows and "original" in rows:
        records = [
            {
                "original": rows["original"][index],
                selected: rows[selected][index],
            }
            for index in range(len(rows["original"]))
        ]
        comparisons = {
            metric: _paired(records, selected, "original", metric)
            for metric in ("room_f1", "room_iou", "corner_f1")
        }
    report = {
        "status": "complete",
        "scope": f"Raster2Seq {args.eval_set} eight-view dihedral consensus",
        "selection_split": "val" if args.force_strategy is None else "forced_policy",
        "official_test509_used": args.eval_set == "test",
        "runs": {
            name: str(path.relative_to(ROOT))
            for name, (path, _) in paths.items()
        },
        "strategies": summaries,
        "eligible": sorted(eligible),
        "selected": selected,
        "paired_vs_original": comparisons,
        "diagnostics": diagnostics,
        "case_rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "selected": selected,
                "metrics": summaries[selected],
                "eligible": {
                    name: summaries[name] for name in eligible
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
