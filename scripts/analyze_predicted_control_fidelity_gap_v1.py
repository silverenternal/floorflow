"""Decompose a frozen predicted-control validation report before more training."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--room-gate", type=float, default=0.35)
    parser.add_argument("--instance-gate", type=float, default=0.30)
    parser.add_argument("--top-cases", type=int, default=40)
    return parser.parse_args()


def _mean(rows: list[dict], key: str) -> float:
    values = [float(row.get(key, 0.0)) for row in rows]
    return float(np.mean(values)) if values else 0.0


def _rate(rows: list[dict], key: str) -> float:
    return _mean(rows, key) if rows else 0.0


def _count_bin(delta: float) -> str:
    if delta <= -3:
        return "under_major"
    if delta <= -1:
        return "under_minor"
    if delta == 0:
        return "exact"
    if delta <= 2:
        return "over_minor"
    return "over_major"


def _target_count_bin(count: float) -> str:
    if count <= 4:
        return "0-4"
    if count <= 7:
        return "5-7"
    if count <= 10:
        return "8-10"
    return "11+"


def _abs_count_bin(error: float) -> str:
    if error == 0:
        return "0"
    if error <= 2:
        return "1-2"
    if error <= 5:
        return "3-5"
    return "6+"


def _fragment_bin(rate: float) -> str:
    if rate == 0:
        return "0"
    if rate <= 0.10:
        return "0-0.10"
    if rate <= 0.25:
        return "0.10-0.25"
    return ">0.25"


def _ged_bin(value: float) -> str:
    if value <= 8:
        return "<=8"
    if value <= 10:
        return "8-10"
    if value <= 12:
        return "10-12"
    return ">12"


def _group_summary(
    rows: list[dict],
    key_fn: Callable[[dict], str],
    total_cases: int,
    room_gate: float,
    instance_gate: float,
) -> dict[str, dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[key_fn(row)].append(row)
    result = {}
    for name, selected in sorted(groups.items()):
        room = np.asarray(
            [float(row.get("room_macro_iou", 0.0)) for row in selected],
            dtype=np.float64,
        )
        instance = np.asarray(
            [
                float(row.get("instance_instance_f1_at_threshold", 0.0))
                for row in selected
            ],
            dtype=np.float64,
        )
        result[name] = {
            "cases": len(selected),
            "case_share": len(selected) / max(1, total_cases),
            "room_macro_iou_mean": float(room.mean()),
            "instance_f1_mean": float(instance.mean()),
            "instance_precision_mean": _mean(
                selected,
                "instance_instance_precision_at_threshold",
            ),
            "instance_recall_mean": _mean(
                selected,
                "instance_instance_recall_at_threshold",
            ),
            "type_accuracy_mean": _mean(selected, "instance_matched_type_accuracy"),
            "count_mae": _mean(selected, "instance_count_abs_error"),
            "count_delta_mean": _mean(selected, "count_delta"),
            "tiny_component_rate_mean": _mean(
                selected,
                "tiny_room_component_rate",
            ),
            "room_deficit_sum": float(np.maximum(0.0, room_gate - room).sum()),
            "instance_deficit_sum": float(
                np.maximum(0.0, instance_gate - instance).sum()
            ),
            "room_gate_pass_rate": float((room >= room_gate).mean()),
            "instance_gate_pass_rate": float((instance >= instance_gate).mean()),
            "plan_plausible_rate": _rate(selected, "plan_plausible"),
            "functional_watertight_rate": _rate(
                selected,
                "functional_watertight",
            ),
            "wall_continuity_mean": _mean(selected, "wall_continuity_rate"),
            "ged_mean": _mean(selected, "ged"),
        }
    return result


def _cases_to_gate(values: np.ndarray, gate: float) -> int:
    deficit = max(0.0, gate * len(values) - float(values.sum()))
    if deficit == 0.0:
        return 0
    headroom = np.sort(np.maximum(0.0, 1.0 - values))[::-1]
    running = 0.0
    for index, value in enumerate(headroom, start=1):
        running += float(value)
        if running >= deficit:
            return index
    return len(values)


def _decomposition_summary(rows: list[dict]) -> dict:
    detailed = [
        row["room_error_decomposition"]
        for row in rows
        if row.get("room_error_decomposition")
    ]
    if not detailed:
        return {"available": False}
    allocation_keys = (
        "correct_identity_fraction",
        "semantic_mismatch_fraction",
        "missing_room_fraction",
        "spurious_room_fraction",
    )
    allocation = {
        key: float(np.mean([item["allocation"][key] for item in detailed]))
        for key in allocation_keys
    }
    classes: dict[str, list[dict]] = defaultdict(list)
    confusions: Counter[tuple[str, str]] = Counter()
    for item in detailed:
        for row in item["class_rows"]:
            if row["present_in_target"]:
                classes[row["class"]].append(row)
        for row in item["confusion_rows"]:
            if row["target_class"] != row["predicted_class"]:
                confusions[(row["target_class"], row["predicted_class"])] += int(
                    row["pixels"]
                )
    class_rows = []
    for name, selected in classes.items():
        target_pixels = sum(int(row["target_pixels"]) for row in selected)
        predicted_pixels = sum(int(row["predicted_pixels"]) for row in selected)
        true_positive = sum(int(row["true_positive"]) for row in selected)
        union = target_pixels + predicted_pixels - true_positive
        class_rows.append(
            {
                "class": name,
                "cases": len(selected),
                "mean_case_iou": float(np.mean([row["iou"] for row in selected])),
                "dataset_iou": true_positive / max(1, union),
                "missing_prediction_rate": float(
                    np.mean([row["missing_prediction"] for row in selected])
                ),
                "mean_iou_deficit_share": float(
                    np.mean([row["mean_iou_deficit_share"] for row in selected])
                ),
            }
        )
    class_rows.sort(key=lambda row: row["mean_iou_deficit_share"], reverse=True)
    return {
        "available": True,
        "cases": len(detailed),
        "allocation": allocation,
        "fragmentation_gap_mean": _mean(detailed, "fragmentation_gap"),
        "missing_class_count_mean": _mean(detailed, "missing_class_count"),
        "spurious_class_count_mean": _mean(detailed, "spurious_class_count"),
        "class_rows": class_rows,
        "dominant_semantic_confusions": [
            {"target_class": target, "predicted_class": predicted, "pixels": pixels}
            for (target, predicted), pixels in confusions.most_common(30)
        ],
    }


def main() -> None:
    args = _parse_args()
    report = json.loads(args.input.read_text(encoding="utf-8"))
    rows = report["cases"]
    for row in rows:
        row["count_delta"] = (
            float(row.get("instance_predicted_instances", 0.0))
            - float(row.get("instance_target_instances", 0.0))
        )
    room = np.asarray(
        [float(row.get("room_macro_iou", 0.0)) for row in rows],
        dtype=np.float64,
    )
    instance = np.asarray(
        [
            float(row.get("instance_instance_f1_at_threshold", 0.0))
            for row in rows
        ],
        dtype=np.float64,
    )
    result = {
        "schema_version": "predicted-control-fidelity-gap-v1",
        "source_report": str(args.input),
        "cases": len(rows),
        "source_protocol": {
            "split": report.get("split"),
            "validation_cases": report.get("validation_cases"),
            "test_opened": report.get("test_opened"),
            "target_used_at_inference": report.get("target_used_at_inference"),
            "config": report.get("config", {}),
        },
        "gates": {
            "room_macro_iou": args.room_gate,
            "instance_f1": args.instance_gate,
        },
        "overall": {
            "room_macro_iou_mean": float(room.mean()),
            "instance_f1_mean": float(instance.mean()),
            "room_gate_gap": float(max(0.0, args.room_gate - room.mean())),
            "instance_gate_gap": float(
                max(0.0, args.instance_gate - instance.mean())
            ),
            "room_oracle_cases_to_gate": _cases_to_gate(room, args.room_gate),
            "instance_oracle_cases_to_gate": _cases_to_gate(
                instance,
                args.instance_gate,
            ),
            "predicted_instances_total": int(
                sum(row.get("instance_predicted_instances", 0.0) for row in rows)
            ),
            "target_instances_total": int(
                sum(row.get("instance_target_instances", 0.0) for row in rows)
            ),
            "over_count_cases": sum(row["count_delta"] > 0 for row in rows),
            "exact_count_cases": sum(row["count_delta"] == 0 for row in rows),
            "under_count_cases": sum(row["count_delta"] < 0 for row in rows),
            "over_count_extra_instances": int(
                sum(max(0.0, row["count_delta"]) for row in rows)
            ),
            "under_count_missing_instances": int(
                sum(max(0.0, -row["count_delta"]) for row in rows)
            ),
            "count_mae": float(np.mean(np.abs([row["count_delta"] for row in rows]))),
        },
        "strata": {
            "count_delta": _group_summary(
                rows,
                lambda row: _count_bin(row["count_delta"]),
                len(rows),
                args.room_gate,
                args.instance_gate,
            ),
            "target_room_count": _group_summary(
                rows,
                lambda row: _target_count_bin(
                    row.get("instance_target_instances", 0.0)
                ),
                len(rows),
                args.room_gate,
                args.instance_gate,
            ),
            "absolute_count_error": _group_summary(
                rows,
                lambda row: _abs_count_bin(
                    row.get("instance_count_abs_error", 0.0)
                ),
                len(rows),
                args.room_gate,
                args.instance_gate,
            ),
            "tiny_component_rate": _group_summary(
                rows,
                lambda row: _fragment_bin(
                    row.get("tiny_room_component_rate", 0.0)
                ),
                len(rows),
                args.room_gate,
                args.instance_gate,
            ),
            "ged": _group_summary(
                rows,
                lambda row: _ged_bin(row.get("ged", 0.0)),
                len(rows),
                args.room_gate,
                args.instance_gate,
            ),
            "plan_plausible": _group_summary(
                rows,
                lambda row: "pass" if row.get("plan_plausible") else "fail",
                len(rows),
                args.room_gate,
                args.instance_gate,
            ),
        },
        "room_decomposition": _decomposition_summary(rows),
        "worst_room_cases": sorted(
            rows,
            key=lambda row: (
                row.get("room_macro_iou", 0.0),
                row.get("instance_instance_f1_at_threshold", 0.0),
            ),
        )[: args.top_cases],
        "worst_instance_cases": sorted(
            rows,
            key=lambda row: (
                row.get("instance_instance_f1_at_threshold", 0.0),
                row.get("room_macro_iou", 0.0),
            ),
        )[: args.top_cases],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["overall"], indent=2))


if __name__ == "__main__":
    main()
