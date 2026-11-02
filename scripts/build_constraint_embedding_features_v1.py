"""Materialize balanced graph-state features for constraint pretraining."""

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
    action_features,
    control_graph_state,
    hard_feasibility,
    hard_feasibility_baseline,
    load_cached_footprint,
)
from floorflow.data.constraint_candidate_dataset import load_candidate_index
from floorflow.data.constraint_candidates import (
    VIOLATION_FAMILIES,
    CandidateSpec,
    constraint_diagnostics,
    is_hard_legal,
    materialize_candidate,
)
from floorflow.data.control_cache import read_control_cache
from floorflow.data.local_control_edits import LOCAL_EDIT_OPERATIONS
from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.model.constraint_state_encoder import room_family_ids

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
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
    parser.add_argument("--negative-per-group", type=int, default=2)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--limit-groups", type=int)
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/control_contracts/residential_cagn_features_v1",
    )
    return parser.parse_args()


def _record(row: dict) -> ActionRecord:
    return ActionRecord(
        **{
            **row,
            "bbox": tuple(row["bbox"]),
            "control_size": tuple(row["control_size"]),
        }
    )


def _selected_specs(
    specs: tuple[CandidateSpec, ...],
    negative_count: int,
) -> list[CandidateSpec]:
    selected = [spec for spec in specs if spec.violation_family is None]
    negatives = [spec for spec in specs if spec.violation_family is not None]
    offset = int(
        hashlib.sha256(specs[0].group_id.encode("utf-8")).hexdigest()[:8],
        16,
    ) % len(negatives)
    for index in range(min(negative_count, len(negatives))):
        selected.append(negatives[(offset + index) % len(negatives)])
    return selected


_TAXONOMY: FloorplanTaxonomy | None = None
_CACHE_ROOT: Path | None = None
_NEGATIVE_PER_GROUP = 2


def _init_worker(
    taxonomy: FloorplanTaxonomy,
    cache_root: str,
    negative_per_group: int,
) -> None:
    global _TAXONOMY, _CACHE_ROOT, _NEGATIVE_PER_GROUP
    torch.set_num_threads(1)
    _TAXONOMY = taxonomy
    _CACHE_ROOT = Path(cache_root)
    _NEGATIVE_PER_GROUP = int(negative_per_group)


def _materialize_source(
    source_groups: tuple[tuple[int, dict], ...],
) -> dict:
    taxonomy = _TAXONOMY
    cache_root = _CACHE_ROOT
    if taxonomy is None or cache_root is None:
        raise RuntimeError("feature worker is not initialized")
    arrays = defaultdict(list)
    role_counts = Counter()
    family_counts = Counter()
    included_groups = 0
    excluded_groups = 0
    first_record = _record(source_groups[0][1]["record"])
    before = read_control_cache(Path(first_record.source))
    footprint = load_cached_footprint(cache_root, first_record)
    baseline = hard_feasibility_baseline(before, taxonomy, footprint)
    edits = {}
    for group_index, group in source_groups:
        record = _record(group["record"])
        edit = edits.get(record.operation)
        if edit is None:
            edit = LOCAL_EDIT_OPERATIONS[record.operation](
                before,
                taxonomy,
                seed=record.seed,
            )
            edits[record.operation] = edit
        if edit is None:
            raise RuntimeError(f"source edit does not reproduce: {record}")
        specs = tuple(CandidateSpec(**row) for row in group["candidates"])
        group_rows = []
        group_legal = True
        for spec in _selected_specs(specs, _NEGATIVE_PER_GROUP):
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
            hard_flags = hard_feasibility(
                before,
                candidate.controls,
                candidate_record,
                taxonomy,
                footprint=footprint,
                baseline=baseline,
            )
            legal = is_hard_legal(hard_flags, candidate.violation_labels)
            if spec.violation_family is None and not legal:
                group_legal = False
            state = control_graph_state(candidate.controls, taxonomy)
            node_mask = candidate.controls["program"].bool()
            if not bool(node_mask.any()):
                node_mask[0] = True
            row = {
                "node_features": state.node_features.to(torch.float16),
                "adjacency": state.adjacency.to(torch.float16),
                "global_features": state.global_features.to(torch.float16),
                "node_mask": node_mask,
                "operation_ids": (
                torch.tensor(
                    tuple(LOCAL_EDIT_OPERATIONS).index(record.operation),
                    dtype=torch.long,
                )
                ),
                "violation_labels": torch.tensor(
                    [
                        candidate.violation_labels[family]
                        for family in VIOLATION_FAMILIES
                    ],
                    dtype=torch.float32,
                ),
                "legal": torch.tensor(legal, dtype=torch.bool),
                "group_index": torch.tensor(group_index),
                "room_count": torch.tensor(
                    int(candidate.controls["program"].sum())
                ),
                "hard_mask": torch.tensor(
                    [hard_flags[key] for key in sorted(hard_flags)],
                    dtype=torch.bool,
                ),
                "constraint_diagnostics": constraint_diagnostics(
                    before,
                    candidate.controls,
                    candidate_record,
                    taxonomy,
                    footprint,
                ),
                "candidate_hash": torch.tensor(
                    int(
                        hashlib.sha256(
                            spec.candidate_id.encode("utf-8")
                        ).hexdigest()[:15],
                        16,
                    ),
                    dtype=torch.long,
                ),
                "action_features": action_features(
                    before,
                    candidate.controls,
                    candidate_record,
                ),
                "role": spec.role,
                "violation_family": spec.violation_family,
            }
            group_rows.append(row)
        if not group_legal:
            excluded_groups += 1
            continue
        included_groups += 1
        for row in group_rows:
            spec_role = row.pop("role")
            family = row.pop("violation_family")
            for key, value in row.items():
                arrays[key].append(value)
            role_counts[spec_role] += 1
            if family is not None:
                family_counts[family] += 1
    return {
        **{key: torch.stack(values) for key, values in arrays.items()},
        "roles": dict(role_counts),
        "families": dict(family_counts),
        "included_groups": included_groups,
        "excluded_groups": excluded_groups,
    }


def main() -> None:
    args = parse_args()
    taxonomy = FloorplanTaxonomy.read(
        ROOT / "outputs/cache/cubicasa5k_full_v2/taxonomy.json"
    )
    if args.split == "test" and not args.allow_test:
        raise ValueError("test feature materialization requires --allow-test")
    payload = load_candidate_index(
        args.index_root / args.split / "index.json",
        allow_test=args.allow_test,
    )
    groups = (
        payload["groups"][: args.limit_groups]
        if args.limit_groups
        else payload["groups"]
    )
    family_ids = room_family_ids(taxonomy)
    arrays = defaultdict(list)
    role_counts = Counter()
    family_counts = Counter()
    included_groups = 0
    excluded_groups = 0
    bundles: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for group_index, group in enumerate(groups):
        bundles[group["record"]["source"]].append((group_index, group))
    tasks = [tuple(bundle) for bundle in bundles.values()]
    with ProcessPoolExecutor(
        max_workers=max(1, args.workers),
        initializer=_init_worker,
        initargs=(
            taxonomy,
            str(args.cache_root),
            args.negative_per_group,
        ),
    ) as executor:
        for completed, result in enumerate(
            executor.map(_materialize_source, tasks),
            1,
        ):
            for key, value in result.items():
                if key in {"roles", "families"}:
                    target = role_counts if key == "roles" else family_counts
                    target.update(value)
                elif key == "included_groups":
                    included_groups += int(value)
                elif key == "excluded_groups":
                    excluded_groups += int(value)
                else:
                    arrays[key].append(value)
            if completed % 500 == 0:
                print(
                    f"{args.split}: materialized {completed}/{len(tasks)} sources",
                    flush=True,
                )
    arrays = {
        key: torch.cat(values)
        for key, values in arrays.items()
    }
    output = args.output / args.split
    output.mkdir(parents=True, exist_ok=True)
    feature_path = output / "features.pt"
    torch.save(
        {
            **arrays,
            "family_ids": family_ids,
            "violation_families": VIOLATION_FAMILIES,
            "split": args.split,
        },
        feature_path,
    )
    summary = {
        "schema_version": "residential-constraint-embedding-features-v1",
        "split": args.split,
        "feature_path": str(feature_path.relative_to(ROOT))
        if feature_path.is_relative_to(ROOT)
        else str(feature_path),
        "rows": int(arrays["legal"].shape[0]),
        "groups": len(groups),
        "negative_per_group": args.negative_per_group,
        "groups_included": included_groups,
        "groups_excluded_by_hard_envelope": excluded_groups,
        "roles": dict(role_counts),
        "families": dict(family_counts),
        "node_shape": list(arrays["node_features"].shape[1:]),
        "source_audit": "outputs/reports/residential_constraint_action_audit_v1.json",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
