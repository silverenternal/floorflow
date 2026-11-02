#!/usr/bin/env python3
"""Analyze blinded user-study response JSON files."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def bootstrap_rate(values: np.ndarray, seed: int = 41) -> dict[str, float]:
    if not len(values):
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    rng = np.random.default_rng(seed)
    means = np.asarray(
        [
            values[rng.integers(0, len(values), len(values))].mean()
            for _ in range(2000)
        ]
    )
    return {
        "mean": float(values.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--answer-key",
        type=Path,
        default=Path("outputs/user_study_pilot_v2/answer_key.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    answer_key = load(args.answer_key)["cases"]
    key_by_case = {item["case_id"]: item for item in answer_key}
    condition_rows = defaultdict(
        lambda: {"style_fit": [], "legality": [], "confidence": [], "wins": []}
    )
    participant_rows = []
    excluded = []
    style_vs_no_style = []
    for path in args.responses:
        payload = load(path)
        participant = payload.get("participant", {})
        if not participant.get("consent") or not participant.get("attention_ok"):
            excluded.append({"path": str(path), "reason": "consent_or_attention"})
            continue
        participant_style = []
        for case in payload.get("cases", []):
            key = key_by_case.get(case.get("case_id"))
            if key is None or "ratings" not in case:
                continue
            ratings = case["ratings"]
            for slot, condition in key["slot_conditions"].items():
                row = ratings.get(slot, {})
                if not row:
                    continue
                condition_rows[condition]["style_fit"].append(
                    float(row.get("style_fit_1_to_7", 0))
                )
                condition_rows[condition]["legality"].append(
                    float(row.get("legality_pass_fail") == "pass")
                )
                condition_rows[condition]["confidence"].append(
                    float(row.get("confidence_1_to_7", 0))
                )
                condition_rows[condition]["wins"].append(
                    float(case.get("preferred_slot") == slot)
                )
            preferred_condition = key["slot_conditions"].get(
                case.get("preferred_slot", "")
            )
            participant_style.append(
                float(preferred_condition == "style_conditioned")
            )
            if preferred_condition in {"style_conditioned", "no_style"}:
                style_vs_no_style.append(
                    float(preferred_condition == "style_conditioned")
                )
        participant_rows.append(participant_style)
    conditions = {}
    for name, rows in condition_rows.items():
        conditions[name] = {
            "observations": len(rows["style_fit"]),
            "style_fit_mean": float(np.mean(rows["style_fit"])),
            "legality_pass_rate": float(np.mean(rows["legality"])),
            "confidence_mean": float(np.mean(rows["confidence"])),
            "preferred_rate": bootstrap_rate(np.asarray(rows["wins"])),
        }
    report = {
        "status": "analyzed" if participant_rows else "awaiting_valid_responses",
        "responses": len(args.responses),
        "valid_participants": len(participant_rows),
        "excluded": excluded,
        "conditions": conditions,
        "style_vs_no_style_preference": bootstrap_rate(
            np.asarray(style_vs_no_style)
        ),
        "participant_role_counts": Counter(
            load(path).get("participant", {}).get("role")
            for path in args.responses
        ),
        "best_of_n_used": False,
        "selection_used_for_model_training": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
