"""Validation for the residential end-to-end benchmark claim contract."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

CLAIM_STATES = {"matched", "context_only", "not_comparable", "unavailable"}
EXTERNAL_LEADERBOARD_TRACKS = {
    "cubicasa_reconstruction",
    "cubicasa_generation",
    "cubicasa_predicted_control_editing",
    "rplan_generation",
    "instruction_editing",
}


def load_benchmark_config(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("benchmark config must be a JSON object")
    return payload


def validate_benchmark_config(config: Mapping) -> list[str]:
    errors: list[str] = []
    if config.get("scope") != "residential_floorplan_only":
        errors.append("scope must be residential_floorplan_only")
    methods = config.get("methods")
    tracks = config.get("tracks")
    if not isinstance(methods, Mapping) or not methods:
        return errors + ["methods must be a non-empty object"]
    if not isinstance(tracks, Mapping) or not tracks:
        return errors + ["tracks must be a non-empty object"]
    for method_name, method in methods.items():
        if not isinstance(method, Mapping):
            errors.append(f"method {method_name} must be an object")
            continue
        method_tracks = method.get("task_tracks")
        if not isinstance(method_tracks, list) or not method_tracks:
            errors.append(f"method {method_name} needs task_tracks")
            continue
        unknown = sorted(set(method_tracks) - set(tracks))
        if unknown:
            errors.append(f"method {method_name} has unknown tracks: {unknown}")
        if (
            method.get("claim_enabled")
            and method.get("matched_status") != "matched"
            and method.get("matched_status") != "self"
        ):
            errors.append(
                f"method {method_name} enables a claim without matched status"
            )
    if "cubicasa_oracle_control_diagnostic" in tracks:
        diagnostic = tracks["cubicasa_oracle_control_diagnostic"]
        if diagnostic.get("deployable") is not False:
            errors.append("oracle-control diagnostic must be non-deployable")
    forbidden = set(config.get("claim_rules", {}).get("forbidden_claims", []))
    for required in (
        "Raster2Seq editing-aware superiority",
        "CE2EPlan editing comparison",
        "HouseMind complete-method defeat",
    ):
        if required not in forbidden:
            errors.append(f"missing forbidden claim: {required}")
    return errors


def validate_claim_rows(
    config: Mapping,
    rows: list[Mapping],
) -> list[str]:
    errors: list[str] = []
    tracks = config.get("tracks", {})
    methods = config.get("methods", {})
    for row in rows:
        method = str(row.get("method", ""))
        track = str(row.get("track", ""))
        state = row.get("claim_state")
        if method not in methods:
            errors.append(f"unknown method in claim row: {method}")
            continue
        if track not in tracks:
            errors.append(f"unknown track in claim row: {track}")
            continue
        if state not in CLAIM_STATES:
            errors.append(f"invalid claim state for {method}/{track}: {state}")
        if state == "matched":
            if track not in EXTERNAL_LEADERBOARD_TRACKS:
                errors.append(f"matched state used on non-leaderboard track: {track}")
            if row.get("executable") is not True:
                errors.append(f"matched claim is not executable: {method}/{track}")
            if row.get("same_scorer") is not True:
                errors.append(f"matched claim lacks same scorer: {method}/{track}")
        if method == "Raster2Seq" and track == "cubicasa_predicted_control_editing":
            errors.append("Raster2Seq cannot enter a native editing claim row")
        if method == "CE2EPlan" and "editing" in track:
            errors.append("CE2EPlan cannot enter an editing claim row")
    return errors


__all__ = [
    "CLAIM_STATES",
    "EXTERNAL_LEADERBOARD_TRACKS",
    "load_benchmark_config",
    "validate_benchmark_config",
    "validate_claim_rows",
]
