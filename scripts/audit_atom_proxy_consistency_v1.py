"""Audit atom metric ownership and proxy-to-canonical consistency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from floorflow.evaluation.atom_metric_contract import (
    build_atom_metric_registry_report,
    load_metric_pairs,
)
from floorflow.evaluation.atom_proxy_consistency import (
    ConsistencyThresholds,
    build_consistency_report,
    load_measurement_payloads,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FPTODO = ROOT / "fptodo.json"
DEFAULT_PAIRS = ROOT / "configs/atom_metric_pairs_v1.json"
DEFAULT_REGISTRY = ROOT / "outputs/reports/atom_metric_registry_v1.json"
DEFAULT_OUTPUT = ROOT / "outputs/reports/atom_proxy_real_consistency_v1.json"
DEFAULT_TEMPLATE = ROOT / "outputs/reports/atom_proxy_measurement_template_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fptodo", type=Path, default=DEFAULT_FPTODO)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--measurements", nargs="*", type=Path, default=[])
    parser.add_argument("--registry-output", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--template-output", type=Path, default=DEFAULT_TEMPLATE)
    return parser.parse_args()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _measurement_template(pair_path: Path) -> dict[str, object]:
    pairs = load_metric_pairs(pair_path)
    return {
        "schema_version": "atom-proxy-measurements-v1",
        "target_used_at_inference": False,
        "measurements": [
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
                "baseline_checkpoint": "epoch_00",
                "target_used_at_inference": False,
                "rows": [],
                "sensitivity": [],
            }
            for pair in pairs
        ],
    }


def main() -> None:
    args = parse_args()
    plan = json.loads(args.fptodo.read_text(encoding="utf-8"))
    registry = build_atom_metric_registry_report(
        args.fptodo,
        args.pairs,
        root=ROOT,
    )
    _write_json(args.registry_output, registry)
    _write_json(args.template_output, _measurement_template(args.pairs))

    measurements = load_measurement_payloads(args.measurements)
    contract = plan["metric_protocol"]["proxy_real_contract"]
    consistency = build_consistency_report(
        measurements,
        ConsistencyThresholds.from_contract(contract),
    )
    measured_atoms = {row["atom_id"] for row in consistency["measurements"]}
    registry_atoms = set(registry["atoms"])
    consistency["coverage"] = {
        "registered_atoms": len(registry_atoms),
        "measured_atoms": len(measured_atoms),
        "missing_atoms": sorted(registry_atoms - measured_atoms),
        "all_atoms_measured": measured_atoms == registry_atoms,
        "registry_complete": registry["complete"],
    }
    consistency["status"] = (
        "complete"
        if registry["complete"] and measured_atoms == registry_atoms
        else "measurement_incomplete"
    )
    consistency["registry_report"] = str(args.registry_output)
    _write_json(args.output, consistency)
    print(
        json.dumps(
            {
                "registry": str(args.registry_output),
                "output": str(args.output),
                "registry_complete": registry["complete"],
                "measured_atoms": len(measured_atoms),
                "status": consistency["status"],
            },
            sort_keys=True,
        )
    )
    if not registry["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
