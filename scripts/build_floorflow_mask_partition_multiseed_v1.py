"""Aggregate five-seed FloorFlow mask-partition evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
METRICS = (
    "matched_geometric_iou",
    "instance_f1_at_threshold",
    "matched_type_accuracy",
    "count_abs_error",
)
WITH_POLYGON_RUNS = {
    2905: (
        "outputs/train/floorflow_resnet_mask_partition1024_b8_v1/report.json",
        "outputs/reports/floorflow_mask_readout_maskselect_1024_v2.json",
    ),
    19: (
        "outputs/train/floorflow_mask_partition1024_seed19_v1/report.json",
        "outputs/reports/floorflow_mask_readout_seed19_v1.json",
    ),
    29: (
        "outputs/train/floorflow_mask_partition1024_seed29_v1/report.json",
        "outputs/reports/floorflow_mask_readout_seed29_v1.json",
    ),
    37: (
        "outputs/train/floorflow_mask_partition1024_seed37_v1/report.json",
        "outputs/reports/floorflow_mask_readout_seed37_v1.json",
    ),
    43: (
        "outputs/train/floorflow_mask_partition1024_seed43_v1/report.json",
        "outputs/reports/floorflow_mask_readout_seed43_v1.json",
    ),
}
MASK_ONLY_RUNS = {
    seed: (
        path.replace("floorflow_resnet_mask_partition1024_b8_v1", "floorflow_mask_only1024_seed2905_v1")
        if seed == 2905
        else path.replace(
            f"floorflow_mask_partition1024_seed{seed}_v1",
            f"floorflow_mask_only1024_seed{seed}_v1",
        ),
        evaluation.replace(
            f"floorflow_mask_readout_seed{seed}_v1",
            f"floorflow_mask_only_seed{seed}_v1",
        ).replace(
            "floorflow_mask_readout_maskselect_1024_v2",
            "floorflow_mask_only_seed2905_v1",
        ),
    )
    for seed, (path, evaluation) in WITH_POLYGON_RUNS.items()
}
MASK_CORE_RUNS = {
    seed: (
        path.replace(
            "floorflow_resnet_mask_partition1024_b8_v1",
            "floorflow_maskcore1024_seed2905_v1",
        )
        if seed == 2905
        else path.replace(
            f"floorflow_mask_partition1024_seed{seed}_v1",
            f"floorflow_maskcore1024_seed{seed}_v1",
        ),
        evaluation.replace(
            f"floorflow_mask_readout_seed{seed}_v1",
            f"floorflow_maskcore_seed{seed}_v1",
        ).replace(
            "floorflow_mask_readout_maskselect_1024_v2",
            "floorflow_maskcore_seed2905_v1",
        ),
    )
    for seed, (path, evaluation) in WITH_POLYGON_RUNS.items()
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/floorflow_mask_partition_multiseed_v1.json",
    )
    parser.add_argument(
        "--suite",
        choices=("with_polygon", "mask_only", "mask_core"),
        default="with_polygon",
    )
    return parser.parse_args()


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size < 2:
        return {
            "mean": float(array.mean()),
            "sd": 0.0,
            "min": float(array.min()),
            "max": float(array.max()),
            "ci95_low": float(array.mean()),
            "ci95_high": float(array.mean()),
        }
    half_width = stats.t.ppf(0.975, array.size - 1) * array.std(
        ddof=1
    ) / np.sqrt(array.size)
    return {
        "mean": float(array.mean()),
        "sd": float(array.std(ddof=1)),
        "min": float(array.min()),
        "max": float(array.max()),
        "ci95_low": float(array.mean() - half_width),
        "ci95_high": float(array.mean() + half_width),
    }


def main() -> None:
    args = _args()
    runs = {
        "with_polygon": WITH_POLYGON_RUNS,
        "mask_only": MASK_ONLY_RUNS,
        "mask_core": MASK_CORE_RUNS,
    }[args.suite]
    records = []
    missing = []
    for seed, (report_name, evaluation_name) in runs.items():
        report_path = ROOT / report_name
        evaluation_path = ROOT / evaluation_name
        if not report_path.exists():
            missing.append(report_name)
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        best = max(
            report["history"],
            key=lambda row: row["selection_score"],
        )
        record = {
            "seed": seed,
            "report": report_name,
            "best_epoch": int(best["epoch"]),
            "training_mask_partition": {
                key: float(best["val"][key]) for key in METRICS
            },
        }
        if evaluation_path.exists():
            evaluation = json.loads(
                evaluation_path.read_text(encoding="utf-8")
            )
            record["canonical"] = {
                source: {
                    "metrics": evaluation["arms"][source]["metrics"],
                    "successful_cases": evaluation["arms"][source][
                        "successful_cases"
                    ],
                    "token_roundtrip_exact_rate": evaluation["arms"][source][
                        "token_roundtrip_exact_rate"
                    ],
                }
                for source in ("polygon", "mask_partition")
            }
        else:
            missing.append(evaluation_name)
        records.append(record)
    aggregate = {
        "training_mask_partition": {
            key: _summary(
                [
                    record["training_mask_partition"][key]
                    for record in records
                ]
            )
            for key in METRICS
        }
    }
    paired = None
    if len(records) >= 2 and all(
        "canonical" in record for record in records
    ):
        aggregate["canonical"] = {}
        for source in ("polygon", "mask_partition"):
            aggregate["canonical"][source] = {
                "metrics": {
                    key: _summary(
                        [
                            record["canonical"][source]["metrics"][key]
                            for record in records
                        ]
                    )
                    for key in METRICS
                },
                "assembly_success_mean": float(
                    np.mean(
                        [
                            record["canonical"][source]["successful_cases"]
                            for record in records
                        ]
                    )
                ),
                "assembly_success_min": int(
                    min(
                        record["canonical"][source]["successful_cases"]
                        for record in records
                    )
                ),
                "token_roundtrip_mean": float(
                    np.mean(
                        [
                            record["canonical"][source][
                                "token_roundtrip_exact_rate"
                            ]
                            for record in records
                        ]
                    )
                ),
                "token_roundtrip_min": float(
                    min(
                        record["canonical"][source][
                            "token_roundtrip_exact_rate"
                        ]
                        for record in records
                    )
                ),
            }
        paired = {}
        for key in METRICS:
            differences = [
                record["canonical"]["mask_partition"]["metrics"][key]
                - record["canonical"]["polygon"]["metrics"][key]
                for record in records
            ]
            paired[key] = {
                "differences": differences,
                "summary": _summary(differences),
                "paired_t_p": float(
                    stats.ttest_rel(
                        [
                            record["canonical"]["mask_partition"]["metrics"][key]
                            for record in records
                        ],
                        [
                            record["canonical"]["polygon"]["metrics"][key]
                            for record in records
                        ],
                    ).pvalue
                ),
            }
    payload = {
        "schema": f"floorflow-{args.suite}-multiseed-v1",
        "suite": args.suite,
        "seeds": [record["seed"] for record in records],
        "runs": records,
        "aggregate": aggregate,
        "paired_mask_minus_polygon": paired,
        "ci_method": "two-sided 95% Student-t interval across seeds",
        "missing": missing,
        "test_opened": False,
        "claim_boundary": (
            "RPLAN validation only. Not a test, leaderboard, or matched "
            "comparison to the pretrained CE2EPlan anchor."
        ),
    }
    if args.suite in ("mask_only", "mask_core"):
        baseline_path = (
            ROOT / "outputs/reports/floorflow_mask_partition_multiseed_v1.json"
        )
        if baseline_path.exists():
            baseline = json.loads(
                baseline_path.read_text(encoding="utf-8")
            )
            baseline_by_seed = {
                record["seed"]: record for record in baseline["runs"]
            }
            cross_suite = {}
            for key in METRICS:
                paired_values = [
                    (
                        record["canonical"]["mask_partition"]["metrics"][key],
                        baseline_by_seed[record["seed"]]["canonical"][
                            "mask_partition"
                        ]["metrics"][key],
                    )
                    for record in records
                    if record["seed"] in baseline_by_seed
                    and "canonical" in record
                ]
                differences = [
                    candidate - baseline_value
                    for candidate, baseline_value in paired_values
                ]
                if len(differences) >= 2:
                    cross_suite[key] = {
                        "differences": differences,
                        "summary": _summary(differences),
                        "paired_t_p": float(
                            stats.ttest_rel(
                                [item[0] for item in paired_values],
                                [item[1] for item in paired_values],
                            ).pvalue
                        ),
                    }
            payload["paired_mask_only_minus_polygon_auxiliary"] = {
                "baseline": str(baseline_path.relative_to(ROOT)),
                "metrics": cross_suite,
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "seeds": payload["seeds"],
                "missing": len(missing),
                "pairing_complete": paired is not None,
            }
        )
    )


if __name__ == "__main__":
    main()
