#!/usr/bin/env python3
"""Audit chip atom proxy/canonical consistency with the shared contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from floorflow.chip_flow.training_runtime import write_json_atomic
from floorflow.evaluation.atom_proxy_consistency import (
    ConsistencyThresholds,
    build_consistency_report,
    load_measurement_payloads,
)


def _atom_map(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    atoms = payload.get("atoms")
    if not isinstance(atoms, list):
        raise TypeError("chip atom config must contain an atoms list")
    return {str(atom["id"]): atom for atom in atoms}


def _registry(atoms: dict[str, dict], measurements) -> dict:
    bound = {}
    for measurement in measurements:
        bound.setdefault(measurement.atom_id, []).append(measurement.id)
    rows = []
    for atom_id, atom in atoms.items():
        rows.append(
            {
                "atom_id": atom_id,
                "module": atom.get("module"),
                "paths": atom.get("paths", []),
                "self_eval": atom.get("self_eval", []),
                "proxy_metrics": atom.get("proxy_metrics", []),
                "real_metrics": atom.get("real_metrics", []),
                "measurement_ids": bound.get(atom_id, []),
                "status": "bound" if atom_id in bound else "awaiting_binding",
            }
        )
    unknown = sorted(set(bound) - set(atoms))
    if unknown:
        raise ValueError(f"measurements reference unknown atoms: {unknown}")
    return {
        "version": "chip-atom-metric-registry-v1",
        "atom_count": len(rows),
        "bound_atom_count": sum(row["status"] == "bound" for row in rows),
        "atoms": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--todo", type=Path, default=Path("chipstodo.json"))
    parser.add_argument(
        "--atoms",
        type=Path,
        default=Path("configs/chip_sota_metric_ownership_v1.json"),
    )
    parser.add_argument("--measurements", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--registry-output",
        type=Path,
        default=Path("outputs/reports/chip_atom_metric_registry_v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reports/chip_proxy_real_consistency_v1.json"),
    )
    args = parser.parse_args()
    atoms = _atom_map(args.atoms)
    measurements = load_measurement_payloads(args.measurements)
    registry = _registry(atoms, measurements)
    report = build_consistency_report(measurements, ConsistencyThresholds())
    report["todo"] = str(args.todo)
    report["registry"] = str(args.registry_output)
    report["coverage"] = {
        "registered_atoms": len(atoms),
        "measured_atoms": len({item.atom_id for item in measurements}),
        "bound_atom_count": registry["bound_atom_count"],
        "all_atoms_bound": registry["bound_atom_count"] == len(atoms),
    }
    report["status"] = (
        "complete"
        if report["coverage"]["all_atoms_bound"]
        and report["summary"]["failed"] == 0
        else "measurement_incomplete"
    )
    write_json_atomic(args.registry_output, registry)
    write_json_atomic(args.output, report)
    print(
        json.dumps(
            {
                "bound_atoms": registry["bound_atom_count"],
                "measurements": len(measurements),
                "passed": report["summary"]["passed"],
                "failed": report["summary"]["failed"],
                "status": report["status"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
