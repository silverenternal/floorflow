"""Aggregate matched room-polygon and shared-owner wall readouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT / "outputs/reports/floorflow_sharedboundary_fiveseed_512_v1.json"
)
METRIC_KEYS = (
    "room_iou",
    "instance_f1",
    "node_f1",
    "edge_overlap",
    "ged",
    "validity",
    "conditional_watertightness",
    "dangling_endpoint_rate",
    "wall_f1",
    "wall_iou",
    "opening_f1",
)
LOWER_IS_BETTER = {"ged", "dangling_endpoint_rate"}


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


def _metrics(path: Path) -> dict[str, float]:
    data = _read(path)["metrics"]
    return {
        key: float(data[key])
        for key in (
            *METRIC_KEYS,
            "wall_connector_count",
            "wall_count",
            "wall_connector_length_mean",
            "wall_connector_length_max",
            "wall_connector_length_ratio",
        )
        if key in data
    }


def _statistics(
    control: list[float],
    candidate: list[float],
    *,
    higher_is_better: bool,
) -> dict[str, float | int]:
    control_values = np.asarray(control, dtype=np.float64)
    candidate_values = np.asarray(candidate, dtype=np.float64)
    differences = candidate_values - control_values
    orientation = 1.0 if higher_is_better else -1.0
    good = orientation * differences
    nonzero = good[good != 0.0]
    if nonzero.size:
        ranks = stats.rankdata(np.abs(nonzero))
        effect = float(np.sum(np.sign(nonzero) * ranks) / np.sum(ranks))
        p_value = float(
            stats.wilcoxon(
                nonzero,
                alternative="two-sided",
                method="auto",
            ).pvalue
        )
    else:
        effect = 0.0
        p_value = 1.0
    count = len(differences)
    delta_mean = float(np.mean(differences))
    standard_error = (
        float(np.std(differences, ddof=1) / np.sqrt(count))
        if count > 1
        else 0.0
    )
    critical = (
        float(stats.t.ppf(0.975, count - 1)) if count > 1 else 0.0
    )
    return {
        "control_mean": float(np.mean(control_values)),
        "candidate_mean": float(np.mean(candidate_values)),
        "delta_mean": delta_mean,
        "delta_ci95_low": delta_mean - critical * standard_error,
        "delta_ci95_high": delta_mean + critical * standard_error,
        "delta_min": float(np.min(differences)),
        "delta_max": float(np.max(differences)),
        "improved_seeds": int(np.count_nonzero(good > 0.0)),
        "wilcoxon_p": p_value,
        "rank_biserial_good": effect,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[19, 29, 37, 43, 2905])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--wall-source",
        choices=("shared_network", "shared_graph"),
        default="shared_network",
    )
    args = parser.parse_args()
    report_root = ROOT / "outputs/reports"
    candidate_name = (
        "sharednetwork"
        if args.wall_source == "shared_network"
        else "sharedgraph"
    )
    runs = []
    for seed in args.seeds:
        control_path = _latest(
            [
                report_root
                / f"floorflow_sharedboundary_seed{seed}_structure_roomwalls_512_v1.json",
                report_root
                / f"floorflow_boundarygraph4096_seed{seed}_structure_adaptivebridge_512_v1.json",
            ]
        )
        candidate_path = _latest(
            [
                report_root
                / f"floorflow_sharedboundary_seed{seed}_structure_{candidate_name}_512_v{version}.json"
                for version in (4, 3, 2, 1)
            ]
        )
        control = _metrics(control_path)
        candidate = _metrics(candidate_path)
        shared_keys = [key for key in METRIC_KEYS if key in control and key in candidate]
        runs.append(
            {
                "seed": seed,
                "control_report": str(control_path),
                "candidate_report": str(candidate_path),
                "control": control,
                "candidate": candidate,
                "delta": {
                    key: candidate[key] - control[key] for key in shared_keys
                },
            }
        )
    shared_keys = [
        key
        for key in METRIC_KEYS
        if all(key in run["control"] and key in run["candidate"] for run in runs)
    ]
    payload = {
        "schema": "floorflow-shared-boundary-screen-v1",
        "seeds": args.seeds,
        "readout": {
            "quality": "adaptive_fill",
            "control_walls": "clipped room polygons",
            "candidate_walls": (
                "owner 1-cells merged into continuous simplified boundary polylines"
                if args.wall_source == "shared_network"
                else "owner 1-cells with bounded endpoint closure"
            ),
            "closure_tolerance_px": 2.0,
            "maximum_closure_gap_px": 30.0,
        },
        "candidate_diagnostics": {
            "wall_connector_count_mean": float(
                np.mean(
                    [
                        run["candidate"]["wall_connector_count"]
                        for run in runs
                    ]
                )
            ),
            "wall_connector_count_min": float(
                np.min(
                    [
                        run["candidate"]["wall_connector_count"]
                        for run in runs
                    ]
                )
            ),
            "wall_connector_count_max": float(
                np.max(
                    [
                        run["candidate"]["wall_connector_count"]
                        for run in runs
                    ]
                )
            ),
            "connector_fraction_of_walls_mean": float(
                np.mean(
                    [
                        run["candidate"]["wall_connector_count"]
                        / run["candidate"]["wall_count"]
                        for run in runs
                    ]
                )
            ),
            "wall_connector_length_mean_px": float(
                np.mean(
                    [
                        run["candidate"]["wall_connector_length_mean"]
                        for run in runs
                    ]
                )
            ),
            "wall_connector_length_max_px": float(
                np.max(
                    [
                        run["candidate"]["wall_connector_length_max"]
                        for run in runs
                    ]
                )
            ),
            "wall_connector_length_ratio_mean": float(
                np.mean(
                    [
                        run["candidate"]["wall_connector_length_ratio"]
                        for run in runs
                    ]
                )
            ),
        },
        "runs": runs,
        "summary": {
            key: _statistics(
                [run["control"][key] for run in runs],
                [run["candidate"][key] for run in runs],
                higher_is_better=key not in LOWER_IS_BETTER,
            )
            for key in shared_keys
        },
        "test_opened": False,
        "claim_boundary": (
            "Matched RPLAN validation readout screen. Wall geometry is derived "
            "from the predicted owner cell complex; openings remain "
            "condition-derived. No external SOTA or test claim."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
