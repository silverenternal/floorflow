"""Join typed proxy rows with canonical CAD-layer evaluation reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from floorflow.evaluation.atom_metric_contract import load_metric_pairs

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAIRS = ROOT / "configs/atom_metric_pairs_v1.json"
ALIASES = {
    "fpeval_opening_f1": "vector_opening_f1",
    "fpeval_wall_f1": "vector_wall_f1",
    "furniture_vector_f1": "furniture_vector_f1",
    "edge_overlap": "housemind_edge_overlap",
    "node_f1": "housemind_node_f1",
    "ged": "housemind_ged",
}
CROSS_SOURCE_ATOMS = {"ST-01", "OP-01", "FU-01", "GR-01"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proxy-rows", nargs="*", type=Path, default=[])
    parser.add_argument("--proxy-report-dir", type=Path)
    parser.add_argument("--canonical-dir", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2905)
    return parser.parse_args()


def _jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _canonical_rows(path: Path) -> dict[tuple[str, str], dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    checkpoint = path.stem
    return {
        (checkpoint, str(row["case"])): row
        for row in payload["cases"]
    }


def _contrast_sensitivity(rows: list[dict], pair) -> list[dict]:
    by_checkpoint: dict[str, list[dict]] = {}
    for row in rows:
        by_checkpoint.setdefault(row["checkpoint"], []).append(row)
    if len(by_checkpoint) < 2:
        return []
    baseline_name = min(by_checkpoint)
    baseline = by_checkpoint[baseline_name]
    baseline_proxy = sum(row["proxy"] for row in baseline) / len(baseline)
    baseline_canonical = sum(row["canonical"] for row in baseline) / len(baseline)
    for checkpoint in sorted(by_checkpoint)[1:]:
        values = by_checkpoint[checkpoint]
        proxy = sum(row["proxy"] for row in values) / len(values)
        canonical = sum(row["canonical"] for row in values) / len(values)
        proxy_better = (
            proxy < baseline_proxy
            if pair.proxy_direction == "minimize"
            else proxy > baseline_proxy
        )
        canonical_better = (
            canonical < baseline_canonical
            if pair.canonical_direction == "minimize"
            else canonical > baseline_canonical
        )
        if proxy_better and canonical_better:
            return [
                {
                    "case_id": "cohort_mean",
                    "attack": f"checkpoint_contrast:{checkpoint}",
                    "proxy_baseline": baseline_proxy,
                    "proxy_attack": proxy,
                    "canonical_baseline": baseline_canonical,
                    "canonical_attack": canonical,
                }
            ]
    return []


def main() -> None:
    args = parse_args()
    proxy_rows = [
        row for path in args.proxy_rows for row in _jsonl(path)
    ]
    canonical = {}
    for path in sorted(args.canonical_dir.glob("epoch_*.json")):
        canonical.update(_canonical_rows(path))
    proxy_reports = {}
    if args.proxy_report_dir is not None:
        for path in sorted(args.proxy_report_dir.glob("epoch_*.json")):
            proxy_reports.update(_canonical_rows(path))
    pairs = load_metric_pairs(args.pairs)
    measurements = []
    for pair in pairs:
        if pair.atom_id not in CROSS_SOURCE_ATOMS:
            continue
        rows = []
        canonical_name = ALIASES.get(
            pair.canonical_metric,
            pair.canonical_metric,
        )
        if proxy_reports:
            sources = (
                (
                    checkpoint,
                    case_id,
                    report,
                    canonical[(checkpoint, case_id)],
                )
                for (checkpoint, case_id), report in proxy_reports.items()
                if (checkpoint, case_id) in canonical
            )
            for checkpoint, case_id, proxy, target in sources:
                if pair.proxy_metric not in proxy:
                    continue
                if canonical_name not in target:
                    continue
                rows.append(
                    {
                        "seed": args.seed,
                        "checkpoint": checkpoint,
                        "case_id": case_id,
                        "proxy": float(proxy[pair.proxy_metric]),
                        "canonical": float(target[canonical_name]),
                    }
                )
        else:
            for row in proxy_rows:
                proxy = row.get("proxy_metrics", {})
                target = canonical.get(
                    (str(row["checkpoint"]), str(row["case_id"]))
                )
                if pair.proxy_metric not in proxy or target is None:
                    continue
                if canonical_name not in target:
                    continue
                rows.append(
                    {
                        "seed": row["seed"],
                        "checkpoint": row["checkpoint"],
                        "case_id": row["case_id"],
                        "proxy": float(proxy[pair.proxy_metric]),
                        "canonical": float(target[canonical_name]),
                    }
                )
        if not rows:
            continue
        checkpoints = sorted({row["checkpoint"] for row in rows})
        measurements.append(
            {
                "id": pair.id,
                "atom_id": pair.atom_id,
                "proxy_metric": pair.proxy_metric,
                "canonical_metric": pair.canonical_metric,
                "proxy_direction": pair.proxy_direction,
                "canonical_direction": pair.canonical_direction,
                "proxy_transform": pair.proxy_transform,
                "canonical_transform": pair.canonical_transform,
                "proxy_tolerance": pair.proxy_tolerance,
                "canonical_tolerance": pair.canonical_tolerance,
                "baseline_checkpoint": checkpoints[0],
                "target_used_at_inference": False,
                "rows": rows,
                "sensitivity": _contrast_sensitivity(rows, pair),
            }
        )
    payload = {
        "schema_version": "atom-proxy-measurements-v1",
        "target_used_at_inference": False,
        "measurements": measurements,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "measurements": len(measurements),
                "rows": sum(len(item["rows"]) for item in measurements),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
