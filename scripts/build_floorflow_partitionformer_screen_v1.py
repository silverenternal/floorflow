"""Aggregate matched independent-mask and joint-partition screens."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT / "outputs/reports/floorflow_partitionformer_twoseed_v1.json"
)
QUALITY_KEYS = (
    "matched_geometric_iou",
    "instance_f1_at_threshold",
    "matched_type_accuracy",
    "count_abs_error",
)
STRUCTURE_KEYS = (
    "room_iou",
    "instance_f1",
    "node_f1",
    "edge_overlap",
    "ged",
    "validity",
    "conditional_watertightness",
    "dangling_endpoint_rate",
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _latest(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError(
        "none of the expected reports exist: "
        + ", ".join(str(path) for path in paths)
    )


def _quality(path: Path) -> dict[str, float]:
    data = _read(path)["arms"]["mask_partition"]["metrics"]
    return {key: float(data[key]) for key in QUALITY_KEYS}


def _structure(path: Path) -> dict[str, float]:
    data = _read(path)["metrics"]
    return {key: float(data[key]) for key in STRUCTURE_KEYS}


def _partition(path: Path) -> dict[str, float]:
    data = _read(path)["summary"]
    return {
        key: float(data[key])
        for key in (
            "joint_partition_overlap_fraction",
            "joint_partition_gap_fraction",
            "owner_confidence_mean",
            "owner_entropy_mean",
        )
    }


def _paired_statistics(
    control: list[float],
    candidate: list[float],
    *,
    higher_is_better: bool,
) -> dict[str, float | int]:
    control_values = np.asarray(control, dtype=np.float64)
    candidate_values = np.asarray(candidate, dtype=np.float64)
    differences = candidate_values - control_values
    orientation = 1.0 if higher_is_better else -1.0
    good_differences = orientation * differences
    nonzero = good_differences[good_differences != 0.0]
    if nonzero.size:
        ranks = stats.rankdata(np.abs(nonzero))
        rank_biserial = float(
            np.sum(np.sign(nonzero) * ranks) / np.sum(ranks)
        )
        wilcoxon_p = float(
            stats.wilcoxon(
                nonzero,
                alternative="two-sided",
                method="auto",
            ).pvalue
        )
    else:
        rank_biserial = 0.0
        wilcoxon_p = 1.0
    sample_count = len(differences)
    standard_error = (
        float(np.std(differences, ddof=1) / np.sqrt(sample_count))
        if sample_count > 1
        else 0.0
    )
    critical = (
        float(stats.t.ppf(0.975, sample_count - 1))
        if sample_count > 1
        else 0.0
    )
    delta_mean = float(np.mean(differences))
    return {
        "control_mean": float(np.mean(control_values)),
        "candidate_mean": float(np.mean(candidate_values)),
        "delta_mean": delta_mean,
        "delta_sd": (
            float(np.std(differences, ddof=1))
            if sample_count > 1
            else 0.0
        ),
        "delta_ci95_low": delta_mean - critical * standard_error,
        "delta_ci95_high": delta_mean + critical * standard_error,
        "delta_min": float(np.min(differences)),
        "delta_max": float(np.max(differences)),
        "improved_seeds": int(np.count_nonzero(good_differences > 0.0)),
        "wilcoxon_p": wilcoxon_p,
        "rank_biserial_good": rank_biserial,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--stage",
        choices=(
            "screen",
            "promotion",
            "scale",
            "boundarygraph",
            "boundarygraph_scale",
            "boundarygraph_scale_adaptive",
        ),
        default="screen",
    )
    args = parser.parse_args()
    runs = []
    for seed in args.seeds:
        root = ROOT / "outputs/reports"
        if args.stage == "screen":
            control_quality_path = (
                root
                / f"floorflow_partitionformer_control256_seed{seed}_canonical_v1.json"
            )
            candidate_quality_path = _latest(
                [
                    root
                    / f"floorflow_partitionformer256_seed{seed}_canonical_v{version}.json"
                    for version in (3, 2, 1)
                ]
            )
            control_structure_path = (
                root
                / f"floorflow_partitionformer_control256_seed{seed}_structure_v1.json"
            )
            candidate_structure_path = _latest(
                [
                    root
                    / f"floorflow_partitionformer256_seed{seed}_structure_v{version}.json"
                    for version in (4, 3, 2, 1)
                ]
            )
            control_partition_path = (
                root
                / f"floorflow_partitionformer_control256_seed{seed}_partition_diag_v2.json"
            )
            candidate_partition_path = (
                root
                / f"floorflow_partitionformer256_seed{seed}_partition_diag_v2.json"
            )
        elif args.stage == "promotion":
            control_quality_path = _latest(
                [
                    root
                    / f"floorflow_partitionformer_control1024_seed{seed}_canonical_v{version}.json"
                    for version in (2, 1)
                ]
                + [
                    root
                    / f"floorflow_cad64_control1024_seed{seed}_canonical_v{version}.json"
                    for version in (3, 2, 1)
                ]
            )
            candidate_quality_path = (
                root
                / f"floorflow_partitionformer_nomask1024_seed{seed}_canonical_v1.json"
            )
            control_structure_path = _latest(
                [
                    root
                    / f"floorflow_partitionformer_control1024_seed{seed}_structure_v{version}.json"
                    for version in (2, 1)
                ]
                + [
                    root
                    / f"floorflow_cad64_control1024_seed{seed}_structure_v{version}.json"
                    for version in (2, 1)
                ]
            )
            candidate_structure_path = (
                root
                / f"floorflow_partitionformer_nomask1024_seed{seed}_structure_v1.json"
            )
            control_partition_path = (
                root
                / f"floorflow_partitionformer_control1024_seed{seed}_partition_diag_v1.json"
            )
            candidate_partition_path = (
                root
                / f"floorflow_partitionformer_nomask1024_seed{seed}_partition_diag_v1.json"
            )
        elif args.stage == "scale":
            control_quality_path = (
                root
                / f"floorflow_partitionformer_control4096_current_seed{seed}_canonical_threshold_v1.json"
            )
            candidate_quality_path = (
                root
                / f"floorflow_partitionformer_ce04096_seed{seed}_canonical_hybrid_threshold_v1.json"
            )
            control_structure_path = _latest(
                [
                    root
                    / f"floorflow_partitionformer_control4096_current_seed{seed}_structure_threshold_v1.json",
                    root
                    / f"floorflow_partitionformer_control4096_current_seed{seed}_structure_v1.json",
                ]
            )
            candidate_structure_path = (
                root
                / f"floorflow_partitionformer_ce04096_seed{seed}_structure_hybrid_threshold_v1.json"
            )
            control_partition_path = (
                root
                / f"floorflow_partitionformer_control4096_current_seed{seed}_partition_diag_v1.json"
            )
            candidate_partition_path = (
                root
                / f"floorflow_partitionformer_ce04096_seed{seed}_partition_diag_v1.json"
            )
        elif args.stage == "boundarygraph":
            control_quality_path = _latest(
                [
                    root
                    / f"floorflow_partitionformer_control1024_current_seed{seed}_canonical_v1.json",
                    root
                    / f"floorflow_partitionformer_control1024_current_seed{seed}_canonical_threshold_v1.json",
                ]
            )
            candidate_quality_path = (
                root
                / f"floorflow_boundarygraph1024_seed{seed}_canonical_hybrid_threshold_v1.json"
            )
            control_structure_path = (
                root
                / f"floorflow_partitionformer_control1024_current_seed{seed}_structure_threshold_v1.json"
            )
            candidate_structure_path = (
                root
                / f"floorflow_boundarygraph1024_seed{seed}_structure_hybrid_threshold_v1.json"
            )
            control_partition_path = (
                root
                / f"floorflow_partitionformer_control1024_current_seed{seed}_partition_diag_v1.json"
            )
            candidate_partition_path = (
                root
                / f"floorflow_boundarygraph1024_seed{seed}_partition_diag_v1.json"
            )
        elif args.stage == "boundarygraph_scale_adaptive":
            control_quality_path = (
                root
                / f"floorflow_partitionformer_control4096_current_seed{seed}_quality_adaptivebridge_v1.json"
            )
            candidate_quality_path = (
                root
                / f"floorflow_boundarygraph4096_seed{seed}_quality_adaptivebridge_v1.json"
            )
            control_structure_path = (
                root
                / f"floorflow_partitionformer_control4096_current_seed{seed}_structure_adaptivebridge_512_v1.json"
            )
            candidate_structure_path = (
                root
                / f"floorflow_boundarygraph4096_seed{seed}_structure_adaptivebridge_512_v1.json"
            )
            control_partition_path = (
                root
                / f"floorflow_partitionformer_control4096_current_seed{seed}_partition_diag_v1.json"
            )
            candidate_partition_path = (
                root
                / f"floorflow_boundarygraph4096_seed{seed}_partition_diag_v1.json"
            )
        else:
            control_quality_path = (
                root
                / f"floorflow_partitionformer_control4096_current_seed{seed}_canonical_threshold_v1.json"
            )
            candidate_quality_path = (
                root
                / f"floorflow_boundarygraph4096_seed{seed}_canonical_hybrid_threshold_v1.json"
            )
            control_structure_path = _latest(
                [
                    root
                    / f"floorflow_partitionformer_control4096_current_seed{seed}_structure_threshold_v1.json",
                    root
                    / f"floorflow_partitionformer_control4096_current_seed{seed}_structure_v1.json",
                ]
            )
            candidate_structure_path = (
                root
                / f"floorflow_boundarygraph4096_seed{seed}_structure_hybrid_threshold_v1.json"
            )
            control_partition_path = (
                root
                / f"floorflow_partitionformer_control4096_current_seed{seed}_partition_diag_v1.json"
            )
            candidate_partition_path = (
                root
                / f"floorflow_boundarygraph4096_seed{seed}_partition_diag_v1.json"
            )
        control = _quality(control_quality_path)
        candidate = _quality(candidate_quality_path)
        control_structure = _structure(control_structure_path)
        candidate_structure = _structure(candidate_structure_path)
        control_partition = (
            _partition(control_partition_path)
            if control_partition_path.exists()
            else {}
        )
        candidate_partition = (
            _partition(candidate_partition_path)
            if candidate_partition_path.exists()
            else {}
        )
        runs.append(
            {
                "seed": seed,
                "control_quality": control,
                "candidate_quality": candidate,
                "quality_delta": {
                    key: candidate[key] - control[key]
                    for key in QUALITY_KEYS
                },
                "control_structure": control_structure,
                "candidate_structure": candidate_structure,
                "structure_delta": {
                    key: candidate_structure[key]
                    - control_structure[key]
                    for key in STRUCTURE_KEYS
                },
                "control_partition": control_partition,
                "candidate_partition": candidate_partition,
            }
        )
    summary = {
        "control_quality_mean": {
            key: float(np.mean([run["control_quality"][key] for run in runs]))
            for key in QUALITY_KEYS
        },
        "candidate_quality_mean": {
            key: float(np.mean([run["candidate_quality"][key] for run in runs]))
            for key in QUALITY_KEYS
        },
        "quality_delta_mean": {
            key: float(np.mean([run["quality_delta"][key] for run in runs]))
            for key in QUALITY_KEYS
        },
        "quality_delta_improved_seeds": {
            key: int(
                sum(
                    (
                        run["quality_delta"][key] < 0.0
                        if key == "count_abs_error"
                        else run["quality_delta"][key] > 0.0
                    )
                    for run in runs
                )
            )
            for key in QUALITY_KEYS
        },
        "structure_delta_mean": {
            key: float(np.mean([run["structure_delta"][key] for run in runs]))
            for key in STRUCTURE_KEYS
        },
        "structure_delta_improved_seeds": {
            key: int(
                sum(
                    (
                        run["structure_delta"][key] < 0.0
                        if key in {"ged", "dangling_endpoint_rate"}
                        else run["structure_delta"][key] > 0.0
                    )
                    for run in runs
                )
            )
            for key in STRUCTURE_KEYS
        },
        "quality_statistics": {
            key: _paired_statistics(
                [run["control_quality"][key] for run in runs],
                [run["candidate_quality"][key] for run in runs],
                higher_is_better=key != "count_abs_error",
            )
            for key in QUALITY_KEYS
        },
        "structure_statistics": {
            key: _paired_statistics(
                [run["control_structure"][key] for run in runs],
                [run["candidate_structure"][key] for run in runs],
                higher_is_better=key
                not in {"ged", "dangling_endpoint_rate"},
            )
            for key in STRUCTURE_KEYS
        },
        "candidate_joint_overlap_max": (
            max(
                run["candidate_partition"][
                    "joint_partition_overlap_fraction"
                ]
                for run in runs
            )
            if all(run["candidate_partition"] for run in runs)
            else None
        ),
        "candidate_joint_gap_max": (
            max(
                run["candidate_partition"]["joint_partition_gap_fraction"]
                for run in runs
            )
            if all(run["candidate_partition"] for run in runs)
            else None
        ),
    }
    report = {
        "schema": f"floorflow-partitionformer-{args.stage}-v1",
        "seeds": args.seeds,
        "budget": {
            "train_cases": (
                256
                if args.stage == "screen"
                else (
                    4096
                    if args.stage
                    in {
                        "scale",
                        "boundarygraph_scale",
                        "boundarygraph_scale_adaptive",
                    }
                    else 1024
                )
            ),
            "validation_cases": (
                64
                if args.stage == "screen"
                else (
                    512
                    if args.stage
                    in {
                        "scale",
                        "boundarygraph_scale",
                        "boundarygraph_scale_adaptive",
                    }
                    else 256
                )
            ),
            "validation_offset": 256,
            "epochs": 8 if args.stage != "promotion" else 12,
            "batch_size": (
                16
                if args.stage
                in {
                    "scale",
                    "boundarygraph_scale",
                    "boundarygraph_scale_adaptive",
                }
                else 8
            ),
            "nfe": 1,
        },
        "runs": runs,
        "summary": summary,
        "readout": (
            {
                "quality": "adaptive_fill",
                "structure": "adaptive_fill",
                "settings": "confidence 0.5; predicted count; one NFE",
            }
            if args.stage == "boundarygraph_scale_adaptive"
            else None
        ),
        "test_opened": False,
        "claim_boundary": (
            "Internal matched architecture screen; not a test or external "
            "leaderboard claim."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
