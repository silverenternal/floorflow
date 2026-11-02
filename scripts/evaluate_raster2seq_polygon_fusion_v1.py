"""Evaluate validation-locked orientation-aware Raster2Seq polygon fusion."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "third_party/Raster2Seq")]

from datasets import build_dataset
from evaluations.rplan_eval.Evaluator import Evaluator_RPlan

from floorflow.evaluation.raster2seq_polygon_fusion import (
    FusionConfig,
    fuse_selected_view,
)
from scripts.evaluate_raster2seq_dihedral_consensus_v1 import (
    OPENING_IDS,
    load_runs,
    run_paths,
    select_consensus,
)
from scripts.evaluate_raster2seq_identity_refiner_v1 import (
    _case_ground_truth,
    _summary,
)
from scripts.evaluate_raster2seq_tta_consensus_v1 import (
    _case_metrics,
    _paired,
    cluster_items,
)

ARMS = {
    "support3_iou090_selected2": FusionConfig(
        minimum_support=3,
        original_weight=1.0,
        selected_weight=2.0,
        minimum_agreement=0.90,
        simplify_tolerance=0.25,
    ),
    "support5_iou085_selected3": FusionConfig(
        minimum_support=5,
        original_weight=1.0,
        selected_weight=3.0,
        minimum_agreement=0.85,
        simplify_tolerance=0.25,
    ),
    "support8_iou080_selected3": FusionConfig(
        minimum_support=8,
        original_weight=1.0,
        selected_weight=3.0,
        minimum_agreement=0.80,
        simplify_tolerance=0.25,
    ),
}
NONINFERIORITY_MARGIN = 0.001


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=ROOT / "outputs/cache/raster2seq_cc5k_val_refined_v1",
    )
    parser.add_argument("--eval-set", choices=("val", "test"), default="val")
    parser.add_argument("--run-prefix", default="val503")
    parser.add_argument("--suffix", default="alpha075_v1")
    parser.add_argument("--force-strategy", choices=tuple(ARMS))
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/raster2seq_polygon_fusion_val_v1.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = run_paths(args.run_prefix, args.suffix, args.eval_set)
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
        tuple(dict.fromkeys((args.force_strategy,)))
        if args.force_strategy
        else tuple(ARMS)
    )
    rows = {
        "original": [],
        "count_consensus": [],
        **{strategy: [] for strategy in strategies},
    }
    for index in range(len(dataset)):
        record = dataset[index]
        image_id = int(record["image_id"])
        runs = load_runs(paths, image_id, 256, 256)
        clusters = cluster_items(runs)
        ground_truth = _case_ground_truth(record)
        selected_view = select_consensus(runs, 2)
        candidates = {
            "original": runs["original"],
            "count_consensus": runs[selected_view],
        }
        candidates.update(
            {
                strategy: fuse_selected_view(
                    runs,
                    selected_view,
                    clusters,
                    OPENING_IDS,
                    ARMS[strategy],
                )
                for strategy in strategies
            }
        )
        for strategy, items in candidates.items():
            rows[strategy].append(
                {
                    "image_id": image_id,
                    **_case_metrics(items, ground_truth, evaluator),
                }
            )
    summaries = {
        strategy: _summary(values)
        for strategy, values in rows.items()
    }
    truth = rows["count_consensus"]
    paired = {}
    eligible = []
    for strategy in strategies:
        records = [
            {
                "count_consensus": truth[index],
                strategy: rows[strategy][index],
            }
            for index in range(len(truth))
        ]
        paired[strategy] = {
            metric: _paired(records, strategy, "count_consensus", metric)
            for metric in ("room_f1", "room_iou", "corner_f1")
        }
        candidate = summaries[strategy]
        original = summaries["count_consensus"]
        if (
            candidate["room_f1"]
            >= original["room_f1"] - NONINFERIORITY_MARGIN
            and candidate["room_iou"] > original["room_iou"]
            and candidate["corner_f1"]
            >= original["corner_f1"] - NONINFERIORITY_MARGIN
            and paired[strategy]["room_iou"]["ci_low"] > 0.0
            and paired[strategy]["room_f1"]["ci_low"]
            > -2.0 * NONINFERIORITY_MARGIN
        ):
            eligible.append(strategy)
    selected = args.force_strategy or max(
        eligible,
        key=lambda strategy: (
            summaries[strategy]["room_iou"],
            summaries[strategy]["room_f1"],
        ),
        default="count_consensus",
    )
    report = {
        "status": "complete",
        "scope": f"Raster2Seq {args.eval_set} polygon-fusion consensus",
        "selection_split": "val" if args.force_strategy is None else "forced_policy",
        "official_test509_used": args.eval_set == "test",
        "arms": {
            name: config.__dict__
            for name, config in ARMS.items()
        },
        "selection_rule": {
            "primary": "paired Room-IoU 95% CI lower bound > 0",
            "room_f1_noninferiority_margin": NONINFERIORITY_MARGIN,
            "corner_f1_noninferiority_margin": NONINFERIORITY_MARGIN,
            "paired_room_f1_ci_lower_floor": -2.0 * NONINFERIORITY_MARGIN,
        },
        "strategies": summaries,
        "paired_vs_count_consensus": paired,
        "eligible": eligible,
        "selected": selected,
        "case_rows": rows,
        "boundary": (
            "Fusion changes predicted polygon coordinates only; room counts "
            "come from the selected-view consensus family."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "selected": selected,
                "metrics": summaries[selected],
                "eligible": eligible,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
