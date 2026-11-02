"""Validate and expose the FloorFlow atom-level metric responsibility map."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

SCHEMA_VERSION = "atom-metric-registry-v1"


@dataclass(frozen=True)
class MetricPair:
    """One auditable proxy-to-canonical measurement contract."""

    id: str
    atom_id: str
    proxy_metric: str
    canonical_metric: str
    proxy_direction: str
    canonical_direction: str
    proxy_tolerance: float
    canonical_tolerance: float
    definition: str
    failure_case: str
    proxy_transform: str = "identity"
    canonical_transform: str = "identity"

    def __post_init__(self) -> None:
        for direction in (self.proxy_direction, self.canonical_direction):
            if direction not in {"maximize", "minimize"}:
                raise ValueError(f"invalid metric direction: {direction}")
        for transform in (self.proxy_transform, self.canonical_transform):
            if transform not in {"identity", "absolute"}:
                raise ValueError(f"invalid metric transform: {transform}")
        if self.proxy_tolerance < 0.0 or self.canonical_tolerance < 0.0:
            raise ValueError("metric tolerances must be non-negative")


def _load_object(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def load_metric_pairs(path: str | Path) -> list[MetricPair]:
    """Load pair contracts and reject duplicate IDs or malformed rows."""

    payload = _load_object(Path(path))
    rows = payload.get("pairs")
    if not isinstance(rows, list) or not rows:
        raise ValueError("metric pair config requires a non-empty pairs list")
    pairs = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("metric pair rows must be objects")
        pairs.append(
            MetricPair(
                id=str(row["id"]),
                atom_id=str(row["atom_id"]),
                proxy_metric=str(row["proxy_metric"]),
                canonical_metric=str(row["canonical_metric"]),
                proxy_direction=str(row["proxy_direction"]),
                canonical_direction=str(row["canonical_direction"]),
                proxy_tolerance=float(row["proxy_tolerance"]),
                canonical_tolerance=float(row["canonical_tolerance"]),
                definition=str(row["definition"]),
                failure_case=str(row["failure_case"]),
                proxy_transform=str(row.get("proxy_transform", "identity")),
                canonical_transform=str(
                    row.get("canonical_transform", "identity")
                ),
            )
        )
    identifiers = [pair.id for pair in pairs]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("metric pair IDs must be unique")
    return pairs


def _source_line_count(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


def validate_atom_plan(
    plan: Mapping,
    pairs: list[MetricPair],
    *,
    root: str | Path,
) -> dict[str, list[str]]:
    """Return fail-closed errors and non-blocking warnings."""

    root = Path(root)
    errors: list[str] = []
    warnings: list[str] = []
    atoms = plan.get("atoms")
    contract = plan.get("metric_protocol", {}).get("proxy_real_contract", {})
    contract_id = contract.get("id")
    if not isinstance(atoms, Mapping) or not atoms:
        return {"errors": ["atoms must be a non-empty object"], "warnings": []}
    if contract_id != "PROXY-REAL-V1":
        errors.append("fptodo proxy-real contract is not PROXY-REAL-V1")

    pair_by_atom: dict[str, list[MetricPair]] = {}
    for pair in pairs:
        pair_by_atom.setdefault(pair.atom_id, []).append(pair)
        if pair.atom_id not in atoms:
            errors.append(f"{pair.id}: unknown atom {pair.atom_id}")

    for atom_id, raw_atom in atoms.items():
        if not isinstance(raw_atom, Mapping):
            errors.append(f"{atom_id}: atom must be an object")
            continue
        for field in ("proxy", "canonical", "sensitivity", "paths"):
            values = raw_atom.get(field)
            if not isinstance(values, list) or not values:
                errors.append(f"{atom_id}: missing non-empty {field}")
        if raw_atom.get("proxy_real_contract") != contract_id:
            errors.append(f"{atom_id}: wrong proxy-real contract")
        atom_pairs = pair_by_atom.get(atom_id, [])
        if not atom_pairs:
            errors.append(f"{atom_id}: no proxy-to-canonical pair")
        proxy_metrics = set(raw_atom.get("proxy", []))
        canonical_metrics = set(raw_atom.get("canonical", []))
        for pair in atom_pairs:
            if pair.proxy_metric not in proxy_metrics:
                errors.append(
                    f"{pair.id}: proxy metric {pair.proxy_metric} is not owned"
                )
            if pair.canonical_metric not in canonical_metrics:
                errors.append(
                    f"{pair.id}: canonical metric "
                    f"{pair.canonical_metric} is not owned"
                )
        for raw_path in raw_atom.get("paths", []):
            path = root / str(raw_path)
            if not path.exists():
                errors.append(f"{atom_id}: missing path {raw_path}")
                continue
            if "chip_flow" in path.parts or path.name == "chipstodo.json":
                errors.append(f"{atom_id}: chip path is out of scope: {raw_path}")
            if path.suffix == ".py" and _source_line_count(path) >= 600:
                errors.append(f"{atom_id}: Python file exceeds 599 lines: {raw_path}")

    known_atoms = set(atoms)
    for task in plan.get("tasks", []):
        for atom_id in task.get("atoms", []):
            if atom_id == "all_promoted_residential_atoms":
                continue
            if atom_id not in known_atoms:
                errors.append(f"{task.get('id')}: unknown atom {atom_id}")
        if task.get("status") == "completed":
            errors.append(f"{task.get('id')}: completed tasks must not remain")
    for metric, spec in plan.get("baseline_ledger", {}).items():
        for atom_id in spec.get("owner_atoms", []):
            if atom_id not in known_atoms:
                errors.append(f"{metric}: unknown owner atom {atom_id}")
    for stage in plan.get("model_dataflow", []):
        for atom_id in stage.get("atoms", []):
            if atom_id not in known_atoms:
                errors.append(f"dataflow: unknown atom {atom_id}")

    evaluator_paths = {
        *plan.get("metric_protocol", {}).get("canonical_evaluators", {}).values(),
        *plan.get("metric_protocol", {}).get("reference_contracts", []),
    }
    for raw_path in evaluator_paths:
        path_text = str(raw_path).split(":", maxsplit=1)[0]
        if not (root / path_text).exists():
            errors.append(f"missing canonical metric asset: {path_text}")

    if not plan.get("evidence_ledgers", {}).get("positive"):
        warnings.append("positive evidence ledger is not declared")
    if not plan.get("evidence_ledgers", {}).get("negative"):
        warnings.append("negative evidence ledger is not declared")
    return {"errors": errors, "warnings": warnings}


def build_atom_metric_registry_report(
    plan_path: str | Path,
    pair_path: str | Path,
    *,
    root: str | Path,
) -> dict[str, object]:
    """Build a machine-readable coverage report for all semantic atoms."""

    plan_path = Path(plan_path)
    pair_path = Path(pair_path)
    plan = _load_object(plan_path)
    pairs = load_metric_pairs(pair_path)
    audit = validate_atom_plan(plan, pairs, root=root)
    pair_by_atom: dict[str, list[MetricPair]] = {}
    for pair in pairs:
        pair_by_atom.setdefault(pair.atom_id, []).append(pair)
    atoms = {}
    for atom_id, atom in plan.get("atoms", {}).items():
        atoms[atom_id] = {
            "name": atom.get("name"),
            "paths": atom.get("paths", []),
            "proxy_metrics": atom.get("proxy", []),
            "canonical_metrics": atom.get("canonical", []),
            "sensitivity_cases": atom.get("sensitivity", []),
            "pairs": [asdict(pair) for pair in pair_by_atom.get(atom_id, [])],
            "status": (
                "ready_for_measurement"
                if pair_by_atom.get(atom_id)
                else "missing_pair"
            ),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "source_plan": str(plan_path),
        "source_pairs": str(pair_path),
        "contract": plan.get("metric_protocol", {})
        .get("proxy_real_contract", {})
        .get("id"),
        "counts": {
            "atoms": len(atoms),
            "pairs": len(pairs),
            "tasks": len(plan.get("tasks", [])),
            "baseline_metrics": len(plan.get("baseline_ledger", {})),
        },
        "complete": not audit["errors"],
        "errors": audit["errors"],
        "warnings": audit["warnings"],
        "atoms": atoms,
    }


__all__ = [
    "SCHEMA_VERSION",
    "MetricPair",
    "build_atom_metric_registry_report",
    "load_metric_pairs",
    "validate_atom_plan",
]
