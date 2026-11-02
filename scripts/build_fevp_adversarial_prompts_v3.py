#!/usr/bin/env python3
"""Construct and score 400 adversarial edit-prompt proxy cases."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from floorflow.evaluation.paired_bootstrap import paired_delta
from floorflow.evaluation.statistical_effect_sizes import (
    cliffs_delta,
    rank_biserial_correlation,
)

ROOT = Path(__file__).resolve().parents[1]
ARMS = (
    "typed_edit_graph",
    "dense_cross_attention",
    "no_edit_gating",
    "no_structural_graph",
)
CATEGORY_SPECS = {
    "enclosed_room": {
        "operation": "room_type",
        "prompt": (
            "Modify the enclosed room while keeping every adjacent room fixed."
        ),
        "proxy_definition": (
            "A real room-type edit used as a proxy for an enclosed-room "
            "instruction; neighbors are expected to remain unchanged."
        ),
    },
    "opening_support_conflict": {
        "operation": "opening_delete",
        "prompt": (
            "Delete the opening/support segment while preserving room connectivity."
        ),
        "proxy_definition": (
            "A real opening deletion used as a proxy for a topology-sensitive "
            "wall/opening instruction."
        ),
    },
    "occupied_room_insertion": {
        "operation": "room_add",
        "prompt": (
            "Insert a room where existing room support already occupies the area."
        ),
        "proxy_definition": (
            "A real room-addition edit used as a proxy for an occupied-location "
            "instruction conflict."
        ),
    },
    "ambiguous_global_scope": {
        "operation": "room_remove",
        "prompt": "Make the whole layout more open and coherent.",
        "proxy_definition": (
            "A real room deletion used as a proxy for high-ambiguity global-scope "
            "instruction language."
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-category", type=int, default=100)
    parser.add_argument(
        "--typed-rows",
        type=Path,
        default=ROOT / "outputs/reports/fp_typed_edit_graph_ablation_rows_v1.json",
    )
    parser.add_argument(
        "--delta-rows",
        type=Path,
        default=ROOT / "outputs/reports/fevp_delta_locality_rows_v2.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/fevp_adversarial_prompts_v3.json",
    )
    parser.add_argument(
        "--rows-output",
        type=Path,
        default=ROOT / "outputs/reports/fevp_adversarial_prompt_rows_v3.json",
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=ROOT / "paper/tables/fevp_adversarial_prompts_v3.md",
    )
    return parser.parse_args()


def _key(seed: int, operation: str, name: str) -> str:
    return f"{seed}|{operation}|{name}"


def _stable_order(rows: list[dict]) -> list[dict]:
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(
            _key(
                row["seed"],
                row["operation"],
                row.get("case_id", row.get("name")),
            ).encode("utf-8")
        ).hexdigest(),
    )


def _summary(rows: list[dict], arm: str) -> dict:
    return {
        metric: float(
            np.mean([row["arms"][arm][metric] for row in rows])
        )
        for metric in (
            "outside_pixel_exact",
            "outside_room_exact",
            "locality_violation_rate",
            "validity",
            "delta_iou",
            "node_f1",
            "edge_f1",
            "ged",
        )
    }


def _paired(rows: list[dict], right: str, metric: str, seed: int) -> dict:
    left_values = [
        float(row["arms"]["typed_edit_graph"][metric]) for row in rows
    ]
    right_values = [float(row["arms"][right][metric]) for row in rows]
    differences = np.asarray(left_values) - np.asarray(right_values)
    nonzero = differences[np.abs(differences) > 1.0e-12]
    lower_is_better = metric in {"ged", "locality_violation_rate"}
    test = (
        wilcoxon(
            nonzero,
            alternative="less" if lower_is_better else "greater",
            zero_method="wilcox",
        )
        if nonzero.size
        else None
    )
    result = paired_delta(
        [{"value": value} for value in left_values],
        [{"value": value} for value in right_values],
        "value",
        iterations=2000,
        seed=seed,
    )
    if lower_is_better:
        result["improved"], result["worsened"] = (
            result["worsened"],
            result["improved"],
        )
    rank_biserial = float(rank_biserial_correlation(differences))
    delta = float(cliffs_delta(left_values, right_values))
    direction = -1.0 if lower_is_better else 1.0
    return {
        **result,
        "median_delta": float(np.median(differences)),
        "rank_biserial": rank_biserial,
        "cliffs_delta": delta,
        "oriented_rank_biserial": direction * rank_biserial,
        "oriented_cliffs_delta": direction * delta,
        "oriented_effect_size_positive_favors_typed": True,
        "nonzero_pairs": int(nonzero.size),
        "wilcoxon_p": float(test.pvalue) if test is not None else 1.0,
        "lower_is_better": lower_is_better,
    }


def main() -> None:
    args = parse_args()
    typed_rows = json.loads(args.typed_rows.read_text(encoding="utf-8"))["rows"]
    delta_rows = json.loads(args.delta_rows.read_text(encoding="utf-8"))["rows"]
    delta_by_key = {
        _key(row["seed"], row["operation"], row["case_id"]): row
        for row in delta_rows
    }
    selected = []
    used = set()
    for category, spec in CATEGORY_SPECS.items():
        operation = spec["operation"]
        candidates = [
            row
            for row in typed_rows
            if row["operation"] == operation
            and _key(row["seed"], row["operation"], row["name"]) not in used
        ]
        candidates = _stable_order(candidates)[: args.per_category]
        for row in candidates:
            key = _key(row["seed"], row["operation"], row["name"])
            used.add(key)
            delta = delta_by_key[key]
            selected.append(
                {
                    "prompt_id": f"{category}-{len(selected):03d}",
                    "prompt": spec["prompt"],
                    "proxy_category": category,
                    "proxy_definition": spec["proxy_definition"],
                    "case_id": row["name"],
                    "seed": row["seed"],
                    "operation": row["operation"],
                    "room_count": delta["room_count"],
                    "changed_room_count": delta["changed_room_count"],
                    "edit_area_ratio": delta["edit_area_ratio"],
                    "arms": {
                        arm: {
                            metric: row["arms"][arm][metric]
                            for metric in (
                                "outside_pixel_exact",
                                "outside_room_exact",
                                "locality_violation_rate",
                                "validity",
                                "delta_iou",
                                "node_f1",
                                "edge_f1",
                                "ged",
                            )
                        }
                        for arm in ARMS
                    },
                }
            )
    if not 200 <= len(selected) <= 500:
        raise ValueError(f"adversarial prompt count out of scope: {len(selected)}")
    by_category = defaultdict(list)
    for row in selected:
        by_category[row["proxy_category"]].append(row)
    category_summary = {}
    for category, rows in sorted(by_category.items()):
        category_summary[category] = {
            "cases": len(rows),
            "prompt": CATEGORY_SPECS[category]["prompt"],
            "arms": {
                arm: _summary(rows, arm)
                for arm in ARMS
            },
        }
    paired = {}
    for right_index, right in enumerate(ARMS[1:]):
        for metric_index, metric in enumerate(
            ("outside_pixel_exact", "locality_violation_rate", "ged")
        ):
            paired[f"typed_minus_{right}:{metric}"] = _paired(
                selected,
                right,
                metric,
                seed=91000 + right_index * 10 + metric_index,
            )
    summary = {
        arm: _summary(selected, arm)
        for arm in ARMS
    }
    report = {
        "status": "complete",
        "scope": (
            "Machine-readable adversarial prompt proxies executed through real "
            "matched control-contract rows; not a natural-language understanding benchmark."
        ),
        "prompts": len(selected),
        "per_category": args.per_category,
        "categories": category_summary,
        "summary": summary,
        "paired": paired,
        "checks": {
            "prompt_count_200_to_500": 200 <= len(selected) <= 500,
            "four_categories_complete": len(by_category) == 4,
            "typed_outside_exact_one": (
                summary["typed_edit_graph"]["outside_pixel_exact"] == 1.0
            ),
            "typed_violation_zero": (
                summary["typed_edit_graph"]["locality_violation_rate"] == 0.0
            ),
            "no_gate_exposes_leakage": (
                summary["no_edit_gating"]["outside_pixel_exact"] < 1.0
            ),
            "typed_ged_lower_than_no_gate": (
                summary["typed_edit_graph"]["ged"]
                < summary["no_edit_gating"]["ged"]
            ),
        },
        "boundary": (
            "The prompts are conflict-mode proxies attached to real controls. "
            "They do not evaluate a natural-language parser."
        ),
        "rows": selected,
    }
    failures = [key for key, value in report["checks"].items() if not value]
    if failures:
        report["status"] = "failed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.rows_output.parent.mkdir(parents=True, exist_ok=True)
    args.rows_output.write_text(
        json.dumps({"status": report["status"], "rows": selected}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    table = [
        "# Adversarial Edit-Prompt Proxy Robustness",
        "",
        "| Arm | Outside exact | Violation | Validity | Delta-IoU | Node F1 | Edge F1 | GED |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        values = summary[arm]
        table.append(
            f"| {arm} | {values['outside_pixel_exact']:.6f} | "
            f"{values['locality_violation_rate']:.9f} | "
            f"{values['validity']:.6f} | {values['delta_iou']:.6f} | "
            f"{values['node_f1']:.6f} | "
            f"{values['edge_f1']:.6f} | {values['ged']:.6f} |"
        )
    table.extend(
        [
            "",
            "| Prompt proxy | Cases | Delta-IoU typed | GED typed | Violation typed | GED no-gate |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for category, values in category_summary.items():
        typed = values["arms"]["typed_edit_graph"]
        no_gate = values["arms"]["no_edit_gating"]
        table.append(
            f"| {category}: {values['prompt']} | {values['cases']} | "
            f"{typed['delta_iou']:.6f} | {typed['ged']:.6f} | "
            f"{typed['locality_violation_rate']:.9f} | {no_gate['ged']:.6f} |"
        )
    table.extend(
        [
            "",
            report["boundary"],
            "",
            "The paired tests compare typed against dense, no-gate, and graphless on the same rows.",
            "",
        ]
    )
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text("\n".join(table), encoding="utf-8")
    print(json.dumps({"status": report["status"], **report["checks"]}, indent=2))
    raise SystemExit(0 if report["status"] == "complete" else 1)


__all__ = ["main"]


if __name__ == "__main__":
    main()
