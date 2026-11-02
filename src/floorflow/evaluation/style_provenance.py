"""Fail-closed provenance checks for user-selected floorplan styles."""

from __future__ import annotations

from collections.abc import Iterable

import torch


def audit_style_provenance(
    bank: dict,
    *,
    train_names: Iterable[str],
    val_names: Iterable[str],
    test_names: Iterable[str],
    checkpoint: dict | None = None,
    run_manifest: dict | None = None,
) -> dict:
    """Verify that style labels are train-only and deploy-time explicit."""

    assignments = bank.get("assignments", {})
    train_assignments = set(assignments.get("train", {}))
    train_set = {str(name) for name in train_names}
    val_set = {str(name) for name in val_names}
    test_set = {str(name) for name in test_names}
    checks: dict[str, bool] = {
        "only_train_split_has_assignments": set(assignments) == {"train"},
        "train_assignments_match_train_split": train_assignments == train_set,
        "no_val_assignment_leak": not bool(train_assignments & val_set),
        "no_test_assignment_leak": not bool(train_assignments & test_set),
        "explicit_deploy_time_rule": (
            "user" in str(bank.get("inference_rule", "")).lower()
            or "deploy" in str(bank.get("inference_rule", "")).lower()
        ),
    }
    if checkpoint is not None:
        config = checkpoint.get("config", {})
        checks["style_count_matches_checkpoint"] = int(
            config.get("num_styles", 0)
        ) == int(bank.get("num_styles", 0))
        style_tensors = [
            value
            for key, value in checkpoint.items()
            if "style" in str(key).lower() and torch.is_tensor(value)
        ]
        for section in ("pixel_head", "element_head", "velocity_style_adapter"):
            state = checkpoint.get(section) or {}
            style_tensors.extend(
                value
                for key, value in state.items()
                if "style" in str(key).lower() and torch.is_tensor(value)
            )
        checks["style_weights_finite"] = all(
            bool(torch.isfinite(value).all()) for value in style_tensors
        )
    if run_manifest is not None:
        checks["contour_only_training"] = (
            run_manifest.get("condition_mode") == "contour_only"
            and int(run_manifest.get("num_styles", 0))
            == int(bank.get("num_styles", 0))
        )
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "train_cases": len(train_set),
        "val_cases": len(val_set),
        "test_cases": len(test_set),
        "num_styles": int(bank.get("num_styles", 0)),
        "inference_rule": bank.get("inference_rule"),
        "best_of_n_used": False,
    }


__all__ = ["audit_style_provenance"]
