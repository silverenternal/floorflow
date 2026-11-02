"""Summarize strict-locality typed edit results across five seeds."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from floorflow.evaluation.statistical_effect_sizes import (
    rank_biserial_correlation,
)

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (19, 20, 21, 22, 23)
OPERATIONS = (
    "room_type",
    "room_add",
    "room_remove",
    "opening_delete",
    "furniture_delete",
)
METRICS = (
    "delta_iou",
    "delta_mse",
    "outside_exact_ratio",
    "outside_changed_pixels",
)
FROZEN_RASTER2SEQ = ROOT / "outputs/reports/fp_raster2seq_local_redraw_v1.json"


def report_path(seed: int) -> Path:
    return (
        ROOT
        / "outputs/reports"
        / f"residential_edit_hybrid_typedall_seed{seed}_v1.json"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_raster2seq_rows(report: dict) -> dict[str, dict[str, dict]]:
    rows = {}
    for operation in OPERATIONS:
        operation_rows = report["operations"][operation]["rows"]
        if len(operation_rows) != int(report["operations"][operation]["cases"]):
            raise ValueError(f"frozen Raster2Seq case count mismatch: {operation}")
        by_name = {str(row["name"]): row for row in operation_rows}
        if len(by_name) != len(operation_rows):
            raise ValueError(f"duplicate frozen Raster2Seq names: {operation}")
        rows[operation] = by_name
    return rows


def paired_test(rows: list[dict], left: str, right: str) -> dict:
    by_name: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        values = by_name.setdefault(row["name"], {left: [], right: []})
        values[left].append(float(row[left]["delta_iou"]))
        values[right].append(float(row[right]["delta_iou"]))
    differences = np.asarray(
        [
            np.mean(values[left]) - np.mean(values[right])
            for values in by_name.values()
        ]
    )
    if np.allclose(differences, 0.0):
        statistic, p_value = 0.0, 1.0
    else:
        result = wilcoxon(
            differences,
            zero_method="wilcox",
            alternative="greater",
        )
        statistic, p_value = float(result.statistic), float(result.pvalue)
    return {
        "cases": len(differences),
        "mean_delta": float(differences.mean()),
        "median_delta": float(np.median(differences)),
        "rank_biserial": float(rank_biserial_correlation(differences)),
        "statistic": statistic,
        "p_value": p_value,
    }


def main() -> None:
    reports = {
        seed: json.loads(report_path(seed).read_text(encoding="utf-8"))
        for seed in SEEDS
    }
    raster_report = json.loads(FROZEN_RASTER2SEQ.read_text(encoding="utf-8"))
    frozen_rows = frozen_raster2seq_rows(raster_report)
    operations = {}
    checks = {}
    for operation in OPERATIONS:
        per_seed = {}
        pooled_rows = []
        for seed in SEEDS:
            operation_report = reports[seed]["operations"][operation]
            primary_name = (
                "typed_before_floorflow_after"
                if operation.startswith("room_")
                else "floorflow"
            )
            typed = operation_report["methods"][primary_name]
            frozen = frozen_rows[operation]
            row_names = {str(row["name"]) for row in operation_report["rows"]}
            if row_names != set(frozen):
                missing = len(row_names - set(frozen))
                extra = len(set(frozen) - row_names)
                raise ValueError(
                    f"{operation} seed {seed} frozen Raster2Seq mismatch: "
                    f"{missing} missing, {extra} extra"
                )
            frozen_metrics = {
                "delta_iou": float(
                    np.mean([frozen[name]["delta_iou"] for name in row_names])
                ),
                "delta_mse": float(
                    np.mean([frozen[name]["delta_mse"] for name in row_names])
                ),
                "outside_exact_ratio": float(
                    np.mean(
                        [
                            frozen[name]["outside_exact_ratio"]
                            for name in row_names
                        ]
                    )
                ),
            }
            per_seed[str(seed)] = {
                "cases": operation_report["cases"],
                "primary_method": primary_name,
                "primary_edit": typed,
                "floorflow": operation_report["methods"]["floorflow"],
                "raster2seq_local": frozen_metrics,
                "raster2seq_source": str(FROZEN_RASTER2SEQ.relative_to(ROOT)),
            }
            pooled_rows.extend(
                {
                    "name": row["name"],
                    "primary_edit": row[primary_name],
                    "floorflow": row["floorflow"],
                    "raster2seq_local": {
                        "delta_iou": float(frozen[str(row["name"])]["delta_iou"]),
                        "delta_mse": float(frozen[str(row["name"])]["delta_mse"]),
                        "outside_exact_ratio": float(
                            frozen[str(row["name"])]["outside_exact_ratio"]
                        ),
                    },
                }
                for row in operation_report["rows"]
            )
        metric_summary = {}
        for metric in METRICS:
            values = np.asarray(
                [
                    per_seed[str(seed)]["primary_edit"][metric]
                    for seed in SEEDS
                ]
            )
            metric_summary[metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        operations[operation] = {
            "per_seed": per_seed,
            "primary_edit_summary": metric_summary,
            "seed_variation": {
                "unit": "five repeated execution seeds on identical test503 rows",
                "primary_metric": "delta_iou",
                "range": [
                    metric_summary["delta_iou"]["min"],
                    metric_summary["delta_iou"]["max"],
                ],
                "standard_deviation": metric_summary["delta_iou"]["std"],
                "interpretation": (
                    "Repeated-seed variation only; not an estimate of "
                    "floorplan population variance."
                ),
            },
            "paired_vs_raster2seq_local": paired_test(
                pooled_rows,
                "primary_edit",
                "raster2seq_local",
            ),
        }
        checks[f"{operation}_primary_iou_above_0_608"] = all(
            per_seed[str(seed)]["primary_edit"]["delta_iou"] > 0.608
            for seed in SEEDS
        )
        checks[f"{operation}_primary_outside_exact_one"] = all(
            per_seed[str(seed)]["primary_edit"]["outside_exact_ratio"] == 1.0
            for seed in SEEDS
        )
        checks[f"{operation}_primary_beats_matched_raster2seq"] = (
            operations[operation]["paired_vs_raster2seq_local"]["mean_delta"] > 0.0
            and operations[operation]["paired_vs_raster2seq_local"]["p_value"] < 0.05
        )

    report = {
        "status": "passed" if all(checks.values()) else "not_passed",
        "scope": "CubiCasa5K test503 strict-locality control-gated editing",
        "seeds": list(SEEDS),
        "operations": operations,
        "checks": checks,
        "statistical_unit": {
            "paired_test": "one mean difference per unique floorplan name",
            "bootstrap_or_wilcoxon_duplicates_seed_rows": False,
            "seed_variation": "reported separately",
        },
        "sources": {
            str(report_path(seed).relative_to(ROOT)): sha256(report_path(seed))
            for seed in SEEDS
        }
        | {
            str(FROZEN_RASTER2SEQ.relative_to(ROOT)): sha256(FROZEN_RASTER2SEQ)
        },
        "claim_boundary": (
            "The primary edit uses declared oracle controls. It is a "
            "strict-locality editing result, not an identity reconstruction rank. "
            "Raster2Seq uses the frozen outside-redraw wrapper; it is not a "
            "native editing-aware model."
        ),
    }
    output = ROOT / "outputs/reports/residential_edit_test503_v1.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Residential Edit Results",
        "",
        "| Operation | Delta-IoU mean | Delta-IoU range | Outside exact | Raster2Seq local | Paired p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for operation in OPERATIONS:
        summary = operations[operation]["primary_edit_summary"]["delta_iou"]
        outside = operations[operation]["primary_edit_summary"][
            "outside_exact_ratio"
        ]
        baseline = operations[operation]["per_seed"]["19"]["raster2seq_local"]["delta_iou"]
        paired = operations[operation]["paired_vs_raster2seq_local"]
        lines.append(
            f"| {operation} | {summary['mean']:.6f} | "
            f"{summary['min']:.6f}-{summary['max']:.6f} | "
            f"{outside['min']:.6f}-{outside['max']:.6f} | "
            f"{baseline:.6f} | {paired['p_value']:.3g} |"
        )
    table = ROOT / "paper/tables/residential_edit_fiveseed_v1.md"
    table.parent.mkdir(parents=True, exist_ok=True)
    table.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], **checks}, indent=2))


if __name__ == "__main__":
    main()
