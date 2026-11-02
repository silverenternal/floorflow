"""Evaluate symmetry-consistent polygon consensus on Raster2Seq val503."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from shapely.geometry import Polygon

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT),
    str(ROOT / "third_party/Raster2Seq"),
]

from datasets import build_dataset
from evaluations.rplan_eval.Evaluator import Evaluator_RPlan

from floorflow.evaluation.paired_bootstrap import paired_delta
from floorflow.evaluation.statistical_effect_sizes import (
    cliffs_delta,
    rank_biserial_correlation,
)
from scripts.evaluate_raster2seq_identity_refiner_v1 import (
    _case_ground_truth,
    _metric_row,
    _summary,
)

RUNS = {
    "original": (
        ROOT / "outputs/external_baselines/raster2seq/official_eval_val503_blend_0.75_v1/jsons",
        None,
    ),
    "hflip": (
        ROOT / "outputs/external_baselines/raster2seq/tta_val503_hflip_alpha075_v1/jsons",
        "hflip",
    ),
    "vflip": (
        ROOT / "outputs/external_baselines/raster2seq/tta_val503_vflip_alpha075_v1/jsons",
        "vflip",
    ),
    "rot180": (
        ROOT / "outputs/external_baselines/raster2seq/tta_val503_rot180_alpha075_v1/jsons",
        "rot180",
    ),
}
STRATEGIES = (
    "original",
    "hflip_only",
    "vflip_only",
    "rot180_only",
    "tta_union",
    "tta_supported",
    "pair_rot_w075",
    "pair_rot_w050",
    "pair_rot_w025",
    "case_consistency",
    "count_consensus",
    "count_consensus_t1",
    "count_consensus_t2",
    "count_consensus_t3",
    "medoid_all",
    "consensus_2",
    "consensus_3",
    "consensus_4",
    "consensus_or_original",
    "largest_consensus_2",
)
OPENING_IDS = frozenset({9, 10})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=ROOT / "outputs/cache/raster2seq_cc5k_val_refined_v1",
    )
    parser.add_argument("--eval-set", default="val", choices=("val", "test"))
    parser.add_argument(
        "--original-root",
        type=Path,
        default=RUNS["original"][0],
    )
    parser.add_argument("--hflip-root", type=Path, default=RUNS["hflip"][0])
    parser.add_argument("--vflip-root", type=Path, default=RUNS["vflip"][0])
    parser.add_argument("--rot180-root", type=Path, default=RUNS["rot180"][0])
    parser.add_argument(
        "--force-strategy",
        choices=STRATEGIES,
        default=None,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/raster2seq_tta_consensus_val_v1.json",
    )
    return parser.parse_args()


def polygon(item: dict) -> Polygon:
    points = np.asarray(item["segmentation"], dtype=float).reshape(-1, 2)
    if len(points) < 4:
        return Polygon()
    value = Polygon(points)
    if not value.is_valid:
        value = value.buffer(0)
    if value.geom_type == "MultiPolygon":
        value = max(value.geoms, key=lambda geometry: geometry.area)
    return value if value.geom_type == "Polygon" else Polygon()


def item_iou(left: dict, right: dict) -> float:
    if int(left["category_id"]) != int(right["category_id"]):
        return 0.0
    first = polygon(left)
    second = polygon(right)
    if first.is_empty or second.is_empty:
        return 0.0
    union = first.union(second).area
    return (
        first.intersection(second).area / union
        if union > 0.0
        else 0.0
    )


def inverse(points: list[float], width: int, height: int, transform: str | None):
    if transform is None:
        return list(points)
    output = []
    for index in range(0, len(points), 2):
        x, y = float(points[index]), float(points[index + 1])
        if transform in {"hflip", "rot180"}:
            x = width - 1.0 - x
        if transform in {"vflip", "rot180"}:
            y = height - 1.0 - y
        output.extend([x, y])
    return output


def load_runs(
    image_id: int,
    width: int,
    height: int,
) -> dict[str, list[dict]]:
    output = {}
    for name, (directory, transform) in RUNS.items():
        items = json.loads(
            (directory / f"{image_id:05d}.json").read_text(encoding="utf-8")
        )
        output[name] = [
            {
                **item,
                "segmentation": inverse(
                    np.asarray(item["segmentation"], dtype=float)
                    .reshape(-1)
                    .tolist(),
                    width,
                    height,
                    transform,
                ),
            }
            for item in items
        ]
    return output


def cluster_items(items: dict[str, list[dict]]) -> list[dict]:
    entries = [
        (source, item)
        for source, rows in items.items()
        for item in rows
        if int(item["category_id"]) not in OPENING_IDS
        if not polygon(item).is_empty and polygon(item).area > 0.0
    ]
    entries.sort(
        key=lambda pair: polygon(pair[1]).area,
        reverse=True,
    )
    clusters = []
    for source, item in entries:
        best = None
        best_iou = 0.0
        for cluster in clusters:
            if int(cluster["items"][0][1]["category_id"]) != int(
                item["category_id"]
            ):
                continue
            value = max(item_iou(item, other) for _, other in cluster["items"])
            if value > best_iou:
                best, best_iou = cluster, value
        if best is None or best_iou < 0.5:
            clusters.append({"items": [(source, item)]})
        else:
            best["items"].append((source, item))
    for cluster in clusters:
        cluster["sources"] = {
            source for source, _item in cluster["items"]
        }
    return clusters


def choose_item(cluster: dict, strategy: str) -> dict:
    if strategy == "largest_consensus_2":
        return max(
            (item for _, item in cluster["items"]),
            key=lambda item: polygon(item).area,
        )
    ranked = []
    for source, item in cluster["items"]:
        agreement = float(
            np.mean(
                [item_iou(item, other) for _, other in cluster["items"]]
            )
        )
        ranked.append((agreement, source == "original", polygon(item).area, item))
    return max(ranked, key=lambda row: row[:3])[3]


def resample(points: np.ndarray, count: int = 64) -> np.ndarray:
    closed = np.concatenate([points, points[:1]], axis=0)
    lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    samples = np.linspace(0.0, cumulative[-1], count, endpoint=False)
    result = np.empty((count, 2), dtype=float)
    result[:, 0] = np.interp(samples, cumulative, closed[:, 0])
    result[:, 1] = np.interp(samples, cumulative, closed[:, 1])
    return result


def pair_average(
    items: dict[str, list[dict]],
    secondary: str,
    primary_weight: float,
) -> list[dict]:
    output = []
    for original in items["original"]:
        if int(original["category_id"]) in OPENING_IDS:
            output.append(original)
            continue
        matches = [
            (item_iou(original, candidate), candidate)
            for candidate in items[secondary]
            if int(candidate["category_id"]) == int(original["category_id"])
        ]
        best_iou, best = max(matches, default=(0.0, None), key=lambda row: row[0])
        if best is None or best_iou < 0.25:
            output.append(original)
            continue
        first = resample(
            np.asarray(original["segmentation"], dtype=float).reshape(-1, 2)
        )
        second = resample(
            np.asarray(best["segmentation"], dtype=float).reshape(-1, 2)
        )
        points = primary_weight * first + (1.0 - primary_weight) * second
        output.append({**original, "segmentation": points.tolist()})
    return [
        {**item, "id": index}
        for index, item in enumerate(output)
    ]


def agreement_score(
    items: dict[str, list[dict]],
    source: str,
) -> float:
    others = [name for name in items if name != source]
    scores = []
    for item in items[source]:
        if int(item["category_id"]) in OPENING_IDS:
            continue
        matches = [
            max(
                (
                    item_iou(item, candidate)
                    for candidate in items[other]
                    if int(candidate["category_id"]) == int(item["category_id"])
                ),
                default=0.0,
            )
            for other in others
        ]
        scores.append(float(np.mean(matches)))
    return float(np.mean(scores)) if scores else 0.0


def fused_items(items: dict[str, list[dict]], strategy: str) -> list[dict]:
    if strategy == "original":
        return items["original"]
    if strategy.endswith("_only"):
        return items[strategy.removesuffix("_only")]
    if strategy == "tta_union":
        output = list(items["original"])
        for source in ("hflip", "vflip", "rot180"):
            for item in items[source]:
                if int(item["category_id"]) in OPENING_IDS:
                    continue
                if not any(
                    item_iou(item, other) >= 0.5
                    for other in output
                    if int(item["category_id"]) not in OPENING_IDS
                ):
                    output.append(item)
        return [
            {**item, "id": index}
            for index, item in enumerate(output)
        ]
    if strategy == "tta_supported":
        output = []
        for original in items["original"]:
            if int(original["category_id"]) in OPENING_IDS:
                output.append(original)
                continue
            support = sum(
                any(item_iou(original, item) >= 0.5 for item in items[source])
                for source in ("hflip", "vflip", "rot180")
            )
            if support >= 1:
                output.append(original)
        return [
            {**item, "id": index}
            for index, item in enumerate(output)
        ]
    if strategy.startswith("pair_rot_w"):
        percentage = int(strategy.removeprefix("pair_rot_w"))
        return pair_average(items, "rot180", percentage / 100.0)
    if strategy == "case_consistency":
        scores = {
            source: agreement_score(items, source)
            for source in RUNS
        }
        return items[max(scores, key=scores.get)]
    if strategy.startswith("count_consensus"):
        scores = {
            source: agreement_score(items, source)
            for source in RUNS
        }
        counts = {
            source: sum(
                int(item["category_id"]) not in OPENING_IDS
                for item in source_items
            )
            for source, source_items in items.items()
        }
        median_count = float(np.median(list(counts.values())))
        selected = min(
            counts,
            key=lambda source: (
                abs(counts[source] - median_count),
                source != "original",
                -scores[source],
            ),
        )
        threshold = (
            int(strategy.rsplit("_t", 1)[1])
            if "_t" in strategy
            else 2
        )
        if abs(counts[selected] - median_count) > float(threshold):
            selected = "original"
        return items[selected]
    clusters = cluster_items(items)
    keep = []
    for cluster in clusters:
        support = len(cluster["sources"])
        if strategy == "consensus_or_original":
            keep_cluster = support >= 2 or any(
                source == "original" for source, _ in cluster["items"]
            )
        else:
            minimum_support = {
                "hflip_only": 1,
                "vflip_only": 1,
                "rot180_only": 1,
                "tta_union": 1,
                "tta_supported": 1,
                "medoid_all": 1,
                "consensus_2": 2,
                "largest_consensus_2": 2,
                "consensus_3": 3,
                "consensus_4": 4,
            }[strategy]
            keep_cluster = support >= minimum_support
        if keep_cluster:
            keep.append(cluster)
    chosen = [
        choose_item(cluster, strategy)
        for cluster in keep
    ]
    chosen.extend(
        item
        for item in items["original"]
        if int(item["category_id"]) in OPENING_IDS
    )
    return sorted(
        [
            {**item, "id": index}
            for index, item in enumerate(chosen)
        ],
        key=lambda item: int(item["id"]),
    )


def _f1(row: dict, prefix: str) -> float:
    precision = float(row[f"{prefix}_prec"])
    recall = float(row[f"{prefix}_rec"])
    return (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0.0
        else 0.0
    )


def _case_metrics(items: list[dict], ground_truth, evaluator) -> dict:
    row = _metric_row(items, ground_truth, evaluator)
    return {
        **row,
        "room_f1": _f1(row, "room"),
        "corner_f1": _f1(row, "corner"),
        "room_sem_f1": _f1(row, "room_sem"),
        "window_door_f1": _f1(row, "window_door"),
    }


def _paired(rows: list[dict], left: str, right: str, metric: str) -> dict:
    left_rows = [{"value": row[left][metric]} for row in rows]
    right_rows = [{"value": row[right][metric]} for row in rows]
    stats = paired_delta(
        left_rows,
        right_rows,
        "value",
        iterations=4000,
        seed=261001,
    )
    differences = np.asarray(
        [
            float(a["value"]) - float(b["value"])
            for a, b in zip(left_rows, right_rows, strict=True)
        ]
    )
    return {
        **stats,
        "rank_biserial": float(rank_biserial_correlation(differences)),
        "cliffs_delta": float(
            cliffs_delta(
                [float(row["value"]) for row in left_rows],
                [float(row["value"]) for row in right_rows],
            )
        ),
    }


def main() -> None:
    args = parse_args()
    RUNS.update(
        {
            "original": (args.original_root.resolve(), None),
            "hflip": (args.hflip_root.resolve(), "hflip"),
            "vflip": (args.vflip_root.resolve(), "vflip"),
            "rot180": (args.rot180_root.resolve(), "rot180"),
        }
    )
    strategies = (
        tuple(dict.fromkeys(("original", args.force_strategy)))
        if args.force_strategy
        else STRATEGIES
    )
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
    rows = {strategy: [] for strategy in strategies}
    diagnostics = {}
    for index in range(len(dataset)):
        record = dataset[index]
        image_id = int(record["image_id"])
        runs = load_runs(image_id, 256, 256)
        scores = {
            source: agreement_score(runs, source)
            for source in RUNS
        }
        diagnostics[str(image_id)] = {
            "agreement": scores,
            "selected_source": max(scores, key=scores.get),
            "room_counts": {
                source: sum(
                    int(item["category_id"]) not in OPENING_IDS
                    for item in items
                )
                for source, items in runs.items()
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
        strategy: _summary(values)
        for strategy, values in rows.items()
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
        "scope": f"Raster2Seq {args.eval_set} geometric TTA consensus",
        "selection_split": "val" if args.force_strategy is None else "forced_policy",
        "official_test509_used": args.eval_set == "test",
        "runs": {
            name: str(path.relative_to(ROOT))
            for name, (path, _) in RUNS.items()
        },
        "strategies": summaries,
        "eligible": sorted(eligible),
        "selected": selected,
        "paired_vs_original": comparisons,
        "diagnostics": diagnostics,
        "case_rows": rows,
    }
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
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
