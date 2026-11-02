"""Audit frozen positive, legal, and typed near-violation candidate groups."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

from floorflow.data.constraint_action_dataset import (
    ActionRecord,
    hard_feasibility,
    hard_feasibility_baseline,
    load_cached_footprint,
)
from floorflow.data.constraint_candidate_dataset import load_candidate_index
from floorflow.data.constraint_candidates import (
    VIOLATION_FAMILIES,
    CandidateSpec,
    is_hard_legal,
    materialize_candidate,
)
from floorflow.data.control_cache import read_control_cache
from floorflow.data.local_control_edits import LOCAL_EDIT_OPERATIONS
from floorflow.data.taxonomy import FloorplanTaxonomy

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index-root",
        type=Path,
        default=ROOT / "outputs/control_contracts/residential_cagn_v1",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=ROOT / "outputs/cache/cubicasa5k",
    )
    parser.add_argument("--limit-groups", type=int)
    parser.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 1))
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/residential_constraint_action_audit_v1.json",
    )
    return parser.parse_args()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record(row: dict) -> ActionRecord:
    return ActionRecord(
        **{
            **row,
            "bbox": tuple(row["bbox"]),
            "control_size": tuple(row["control_size"]),
        }
    )


_TAXONOMY: FloorplanTaxonomy | None = None
_CACHE_ROOT: Path | None = None


def _init_worker(taxonomy: FloorplanTaxonomy, cache_root: str) -> None:
    global _TAXONOMY, _CACHE_ROOT
    torch.set_num_threads(1)
    _TAXONOMY = taxonomy
    _CACHE_ROOT = Path(cache_root)


def _audit_source(split_and_groups: tuple[str, tuple[dict, ...]]) -> dict:
    split, groups = split_and_groups
    taxonomy = _TAXONOMY
    cache_root = _CACHE_ROOT
    if taxonomy is None or cache_root is None:
        raise RuntimeError("audit worker is not initialized")
    result = {
        "roles": Counter(),
        "families": Counter(),
        "severity": Counter(),
        "hard_pass": Counter(),
        "legal": Counter(),
        "mask_pass": defaultdict(Counter),
        "operation_roles": defaultdict(Counter),
    }
    first_record = _record(groups[0]["record"])
    source = Path(first_record.source)
    before = read_control_cache(source)
    footprint = load_cached_footprint(cache_root, first_record)
    baseline = hard_feasibility_baseline(before, taxonomy, footprint)
    edits = {}
    for group in groups:
        record = _record(group["record"])
        if record.source != first_record.source:
            raise ValueError("source-group bundle contains multiple controls")
        edit = edits.get(record.operation)
        if edit is None:
            edit = LOCAL_EDIT_OPERATIONS[record.operation](
                before,
                taxonomy,
                seed=record.seed,
            )
            edits[record.operation] = edit
        if edit is None:
            raise RuntimeError(f"candidate source edit does not reproduce: {record}")
        for row in group["candidates"]:
            spec = CandidateSpec(**row)
            candidate = materialize_candidate(
                before,
                edit.controls,
                record,
                spec,
                taxonomy,
                footprint,
            )
            candidate_record = ActionRecord(
                name=record.name,
                split=record.split,
                source=record.source,
                operation=record.operation,
                seed=record.seed,
                bbox=candidate.bbox,
                control_size=record.control_size,
                changed_pixels=candidate.changed_pixels,
            )
            flags = hard_feasibility(
                before,
                candidate.controls,
                candidate_record,
                taxonomy,
                footprint=footprint,
                baseline=baseline,
            )
            legal = is_hard_legal(flags, candidate.violation_labels)
            result["roles"][spec.role] += 1
            result["operation_roles"][spec.operation][spec.role] += 1
            result["severity"][str(spec.severity)] += 1
            result["legal"]["legal" if legal else "illegal"] += 1
            if spec.violation_family is not None:
                result["families"][spec.violation_family] += 1
            if all(flags.values()):
                result["hard_pass"][spec.role] += 1
            for mask_name, passed in flags.items():
                result["mask_pass"][spec.role][mask_name] += int(passed)
    return {
        "split": split,
        "groups": len(groups),
        "source_hash": _file_hash(source),
        "roles": dict(result["roles"]),
        "families": dict(result["families"]),
        "severity": dict(result["severity"]),
        "hard_pass": dict(result["hard_pass"]),
        "legal": dict(result["legal"]),
        "mask_pass": {
            role: dict(counts)
            for role, counts in result["mask_pass"].items()
        },
        "operation_roles": {
            operation: dict(counts)
            for operation, counts in result["operation_roles"].items()
        },
    }


def _audit_split(
    split: str,
    payload: dict,
    taxonomy: FloorplanTaxonomy,
    cache_root: Path,
    limit_groups: int | None,
    workers: int,
) -> dict:
    groups = payload["groups"][:limit_groups] if limit_groups else payload["groups"]
    bundles: dict[str, list[dict]] = defaultdict(list)
    for group in groups:
        bundles[group["record"]["source"]].append(group)
    tasks = [(split, tuple(bundle)) for bundle in bundles.values()]
    role_counts = Counter()
    family_counts = Counter()
    severity_counts = Counter()
    hard_pass = Counter()
    legal_counts = Counter()
    mask_pass = defaultdict(Counter)
    operation_roles = defaultdict(Counter)
    source_hashes = []
    with ProcessPoolExecutor(
        max_workers=max(1, workers),
        initializer=_init_worker,
        initargs=(taxonomy, str(cache_root)),
    ) as executor:
        for completed, result in enumerate(executor.map(_audit_source, tasks), 1):
            role_counts.update(result["roles"])
            family_counts.update(result["families"])
            severity_counts.update(result["severity"])
            hard_pass.update(result["hard_pass"])
            legal_counts.update(result["legal"])
            source_hashes.append(result["source_hash"])
            for role, counts in result["mask_pass"].items():
                mask_pass[role].update(counts)
            for operation, counts in result["operation_roles"].items():
                operation_roles[operation].update(counts)
            if completed % 500 == 0:
                print(f"{split}: audited {completed}/{len(tasks)} sources", flush=True)
    combinations = len(groups) * 9
    hard_mask_pass_rate = {
        role: {
            key: count / max(role_counts[role], 1)
            for key, count in mask_pass[role].items()
        }
        for role in role_counts
    }
    expected_failures = {
        family: family_counts[family] for family in VIOLATION_FAMILIES
    }
    source_digest = hashlib.sha256(
        "".join(sorted(source_hashes)).encode("ascii")
    ).hexdigest()
    return {
        "split": split,
        "groups": len(groups),
        "candidates": combinations,
        "roles": dict(role_counts),
        "families": dict(family_counts),
        "severity": dict(severity_counts),
        "legal": dict(legal_counts),
        "hard_flags_all_pass": dict(hard_pass),
        "hard_mask_pass_rate": hard_mask_pass_rate,
        "operation_roles": {
            operation: dict(counts)
            for operation, counts in operation_roles.items()
        },
        "expected_negative_families": expected_failures,
        "source_digest_sha256": source_digest,
        "source_file_hashes": sorted(source_hashes),
    }


def main() -> None:
    args = parse_args()
    taxonomy = FloorplanTaxonomy.read(
        ROOT / "outputs/cache/cubicasa5k_full_v2/taxonomy.json"
    )
    splits = {}
    for split in ("train", "val"):
        index_path = args.index_root / split / "index.json"
        payload = load_candidate_index(index_path)
        splits[split] = _audit_split(
            split,
            payload,
            taxonomy,
            args.cache_root,
            args.limit_groups,
            args.workers,
        )
    checks = {
        "train_val_only": set(splits) == {"train", "val"},
        "nine_candidates_per_group": all(
            result["candidates"] == 9 * result["groups"]
            for result in splits.values()
        ),
        "positive_hard_pass_rate_at_least_0_98": all(
            result["hard_flags_all_pass"].get("positive", 0)
            / max(result["roles"].get("positive", 0), 1)
            >= 0.98
            for result in splits.values()
        ),
        "legal_alternative_hard_pass_rate_at_least_0_98": all(
            result["hard_flags_all_pass"].get("legal_alternative", 0)
            / max(result["roles"].get("legal_alternative", 0), 1)
            >= 0.98
            for result in splits.values()
        ),
        "every_family_present": all(
            result["families"].get(family, 0) > 0
            for result in splits.values()
            for family in VIOLATION_FAMILIES
        ),
        "hard_mask_recorded_for_every_role": all(
            set(result["hard_mask_pass_rate"])
            == {"positive", "legal_alternative", "near_violation"}
            for result in splits.values()
        ),
    }
    report = {
        "schema_version": "residential-constraint-action-audit-v1",
        "status": "passed" if all(checks.values()) else "failed",
        "scope": "residential_floorplan_only",
        "selection_policy": "train/validation only; test503 is never opened",
        "downstream_policy": (
            "groups with any hard-infeasible positive or legal alternative "
            "are excluded before model training"
        ),
        "splits": splits,
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "checks": checks}, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
