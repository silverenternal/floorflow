"""Audit provenance, balance, hard masks, and diversity of model candidates."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "outputs/control_contracts/model_candidates_v1"
DEFAULT_REPORT = ROOT / "outputs/reports/model_candidate_corpus_audit_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def _entropy(values: list[float], bins: int = 10) -> float:
    if not values:
        return 0.0
    counts = Counter(min(int(value * bins), bins - 1) for value in values)
    probabilities = [count / len(values) for count in counts.values()]
    return float(
        -sum(value * math.log(max(value, 1.0e-12)) for value in probabilities)
        / math.log(bins)
    )


def _normalized(values: list[float]) -> list[float]:
    low = min(values)
    high = max(values)
    if high - low <= 1.0e-12:
        return [0.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def _audit_split(split: str, payload: dict) -> dict:
    roles = Counter()
    families = Counter()
    operations = Counter()
    hard_pass = Counter()
    positive_groups = Counter()
    group_entropy = []
    unique_utilities = []
    near_hard_valid = 0
    near_total = 0
    for group in payload["groups"]:
        operation = group["operation"]
        operations[operation] += 1
        utilities = [
            float(candidate["quality"]["utility"])
            for candidate in group["candidates"]
        ]
        group_entropy.append(_entropy(_normalized(utilities)))
        unique_utilities.append(len({round(value, 6) for value in utilities}))
        has_legal_positive = False
        for candidate in group["candidates"]:
            role = candidate["role"]
            roles[role] += 1
            if candidate["violation_family"] is not None:
                families[candidate["violation_family"]] += 1
            if candidate["hard_feasible"]:
                hard_pass[role] += 1
            if role in {"positive", "legal_alternative"} and candidate["hard_feasible"]:
                has_legal_positive = True
            if role == "near_violation":
                near_total += 1
                near_hard_valid += int(bool(candidate["hard_feasible"]))
        if has_legal_positive:
            positive_groups[operation] += 1
    return {
        "split": split,
        "groups": len(payload["groups"]),
        "candidates": sum(len(group["candidates"]) for group in payload["groups"]),
        "roles": dict(roles),
        "families": dict(families),
        "operations": dict(operations),
        "hard_pass": dict(hard_pass),
        "groups_with_legal_positive": dict(positive_groups),
        "number_of_cases": len({group["name"] for group in payload["groups"]}),
        "mean_utility_entropy": (
            sum(group_entropy) / len(group_entropy) if group_entropy else 0.0
        ),
        "mean_unique_utilities_per_group": (
            sum(unique_utilities) / len(unique_utilities)
            if unique_utilities
            else 0.0
        ),
        "near_violation_hard_valid_rate": (
            near_hard_valid / near_total if near_total else 0.0
        ),
        "source_digest": payload.get("predicted_control_root"),
        "target_used_at_inference": payload.get("target_used_at_inference"),
        "candidate_origin": payload.get("candidate_origin"),
    }


def main() -> None:
    args = parse_args()
    splits = {}
    checks = {}
    for split in ("train", "val"):
        path = args.root / split / "index.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        result = _audit_split(split, payload)
        splits[split] = result
        checks[f"{split}_no_target_at_inference"] = (
            result["target_used_at_inference"] is False
        )
        checks[f"{split}_all_operations"] = set(result["operations"]) == {
            "room_add",
            "room_remove",
            "room_type",
            "opening_delete",
            "furniture_delete",
        }
        checks[f"{split}_legal_positive_every_small_group_set"] = all(
            result["groups_with_legal_positive"].get(operation, 0) > 0
            for operation in result["operations"]
        )
        checks[f"{split}_entropy_above_0_1"] = (
            result["mean_utility_entropy"] >= 0.1
        )
        checks[f"{split}_diverse_utilities"] = (
            result["mean_unique_utilities_per_group"] >= 2.5
        )
        checks[f"{split}_positive_hard_rate_at_least_0_5"] = (
            result["hard_pass"].get("positive", 0)
            / max(result["roles"].get("positive", 0), 1)
            >= 0.5
        )
        checks[f"{split}_typed_near_violations_filtered"] = (
            result["near_violation_hard_valid_rate"] <= 0.01
        )
    report = {
        "schema_version": "model-origin-candidate-audit-v1",
        "status": "passed" if all(checks.values()) else "failed",
        "scope": "residential_floorplan_only_train_validation",
        "candidate_origin": "predicted controls from the real FloorFlow model",
        "oracle_role": "labels/intent scoring only; never candidate input",
        "splits": splits,
        "checks": checks,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"report": str(args.report), "checks": checks}, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
