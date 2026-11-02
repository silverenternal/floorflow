"""Audit interaction-corpus provenance, branching, and legal recovery."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from floorflow.data.interaction_candidate_dataset import (
    CONTINUATION_ROLES,
    INTERACTION_SCHEMA,
    index_sequences,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/reports/residential_interaction_corpus_audit_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT / "outputs/control_contracts/model_interactions_v1",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def _audit_split(root: Path, split: str) -> tuple[dict, list[str]]:
    path = root / split / "index.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    groups = payload["groups"]
    errors = []
    if payload.get("schema_version") != INTERACTION_SCHEMA:
        errors.append("schema mismatch")
    if payload.get("split") != split:
        errors.append("split mismatch")
    if payload.get("target_used_at_inference") is not False:
        errors.append("target-derived controls are not allowed")
    if payload.get("sequence_context", {}).get("feature_dim") != 12:
        errors.append("observable sequence context is missing")
    if any(
        len(candidate.get("sequence_features", [])) != 12
        for group in groups
        for candidate in group["candidates"]
    ):
        errors.append("candidate sequence feature length mismatch")
    sequences = index_sequences(groups)
    roots = [sequence["root"] for sequence in sequences.values()]
    if not roots or any(root is None for root in roots):
        errors.append("missing sequence roots")
    candidate_counts = Counter(
        len(group["candidates"]) for group in groups
    )
    if set(candidate_counts) != {5}:
        errors.append("every group must contain exactly five candidates")
    parent_links = 0
    cross_step_links = 0
    illegal_continuations = 0
    depths = []
    for sequence in sequences.values():
        depths.append(max(sequence["groups"]))
        for parent_id, child in sequence["children"].items():
            parent_step = child["step"] - 1
            parent = sequence["groups"].get(parent_step)
            if parent is None:
                errors.append(f"missing parent step for {parent_id}")
                continue
            parent_row = next(
                (
                    row
                    for row in parent["candidates"]
                    if row["candidate_id"] == parent_id
                ),
                None,
            )
            if parent_row is None:
                errors.append(f"parent candidate missing for {parent_id}")
                continue
            parent_links += 1
            if parent_row["role"] in CONTINUATION_ROLES:
                cross_step_links += 1
            else:
                illegal_continuations += 1
    if illegal_continuations:
        errors.append("near violations must be terminal")
    role_counts = Counter(
        candidate["role"]
        for group in groups
        for candidate in group["candidates"]
    )
    operation_counts = Counter(group["operation"] for group in groups)
    summary = {
        "split": split,
        "sequences": len(sequences),
        "depth_histogram": dict(sorted(Counter(depths).items())),
        "groups": len(groups),
        "candidates": sum(len(group["candidates"]) for group in groups),
        "candidate_count_histogram": dict(sorted(candidate_counts.items())),
        "roles": dict(sorted(role_counts.items())),
        "operations": dict(sorted(operation_counts.items())),
        "parent_links": parent_links,
        "legal_cross_step_links": cross_step_links,
        "illegal_cross_step_links": illegal_continuations,
    }
    return summary, errors


def main() -> None:
    args = parse_args()
    reports = {}
    errors = {}
    for split in ("train", "val"):
        reports[split], errors[split] = _audit_split(args.root, split)
    passed = not any(errors.values())
    payload = {
        "schema_version": "residential-interaction-corpus-audit-v1",
        "status": "passed" if passed else "failed",
        "scope": "residential_floorplan_only",
        "training_split_only": True,
        "test_opened": False,
        "splits": reports,
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "report": (
                    str(args.output.relative_to(ROOT))
                    if args.output.is_relative_to(ROOT)
                    else str(args.output)
                ),
                "status": payload["status"],
            }
        )
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
