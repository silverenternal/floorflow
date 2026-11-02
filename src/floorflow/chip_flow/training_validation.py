"""Validation of chip training scope and numerical options."""

from __future__ import annotations

import argparse
from pathlib import Path

from floorflow.chip_flow.training_scopes import (
    TrainingScopeTransition,
    parse_scope_schedule,
    parse_training_scope,
    validate_atom_isolation_schedule,
)


def validate_atom_metric_objective_schedule(
    args: argparse.Namespace,
    schedule: tuple[TrainingScopeTransition, ...],
) -> None:
    """Require metric objectives to become active inside their owned atom stage."""

    if not schedule:
        return
    windows: dict[str, tuple[int, int]] = {}
    for index, transition in enumerate(schedule):
        end = (
            schedule[index + 1].start
            if index + 1 < len(schedule)
            else transition.start + 1
        )
        for group in transition.scope.groups:
            if group not in {"area", "b2b", "port_route", "boundary", "tail"}:
                continue
            windows[group] = (transition.start, end)
    objectives = (
        ("official_cost", "area"),
        ("grouping_proxy", "b2b"),
        ("routing_proxy", "port_route"),
        ("legalizer_response", "tail"),
        ("endpoint_compatibility", "area"),
    )
    for objective, owner in objectives:
        final_weight = getattr(args, f"{objective}_final_weight", None)
        target = (
            float(final_weight)
            if final_weight is not None
            else float(getattr(args, f"{objective}_weight", 0.0))
        )
        if target <= 0.0:
            continue
        if owner not in windows:
            raise ValueError(
                f"{objective} objective has no isolated {owner} stage"
            )
        start, end = windows[owner]
        delay = int(getattr(args, f"{objective}_delay_steps", 0))
        if delay < start:
            raise ValueError(
                f"{objective} delay {delay} starts before the {owner} stage "
                f"starts at {start}"
            )
        if delay >= end:
            raise ValueError(
                f"{objective} delay {delay} starts after the {owner} stage "
                f"ends at {end}"
            )


def validate_training_scope(args: argparse.Namespace) -> None:
    """Reject accidental full-data runs and invalid training options."""

    if args.train_indices is None and not args.allow_full_data:
        raise ValueError(
            "full-data training requires --allow-full-data; "
            "use --train-indices for exploratory versions"
        )
    if args.batch_size <= 0 or args.grad_accum_steps <= 0:
        raise ValueError("batch size and gradient accumulation must be positive")
    cache_root = getattr(args, "typed_channel_cache_root", None)
    if cache_root is not None and not Path(cache_root).is_dir():
        raise ValueError("typed-channel cache root does not exist")
    if not str(getattr(args, "typed_channel_cache_split", "train")):
        raise ValueError("typed-channel cache split must not be empty")
    training_scope = parse_training_scope(
        str(getattr(args, "training_scope", "all"))
    )
    schedule_text = getattr(args, "training_scope_schedule", None)
    schedule = parse_scope_schedule(
        None if schedule_text is None else str(schedule_text)
    )
    if schedule:
        if training_scope.name != "all":
            raise ValueError(
                "--training-scope must remain all when a schedule is supplied"
            )
        if schedule[0].start != 0:
            raise ValueError("training scope schedule must start at step zero")
    if bool(getattr(args, "require_atom_isolation", False)):
        if not schedule:
            raise ValueError(
                "atom isolation requires a training scope schedule"
            )
        validate_atom_isolation_schedule(schedule)
        validate_atom_metric_objective_schedule(args, schedule)
    if bool(getattr(args, "freeze_existing", False)) and (
        training_scope.name != "all" or schedule
    ):
        raise ValueError("freeze-existing cannot be combined with training scopes")
    if int(getattr(args, "backbone_unfreeze_step", -1) or -1) >= 0 and (
        training_scope.name != "all" or schedule
    ):
        raise ValueError(
            "backbone unfreeze cannot be combined with an isolated training scope"
        )
    if int(getattr(args, "stage_lr_warmup_steps", 25)) < 0:
        raise ValueError("stage learning-rate warmup must be non-negative")
    if bool(getattr(args, "stage_lr_restart", False)) and not schedule:
        raise ValueError("stage LR restart requires a training scope schedule")
    if float(getattr(args, "hierarchy_source_noise", 0.0)) < 0.0:
        raise ValueError("hierarchy source noise must be non-negative")
    if not 0.0 <= float(
        getattr(args, "hierarchy_source_shelf_probability", 0.5)
    ) <= 1.0:
        raise ValueError("source shelf probability must be in [0, 1]")
    source_final = getattr(
        args,
        "hierarchy_source_shelf_probability_final",
        None,
    )
    if source_final is not None and not 0.0 <= float(source_final) <= 1.0:
        raise ValueError("final source shelf probability must be in [0, 1]")
    if int(getattr(args, "hierarchy_source_shelf_probability_steps", 0)) < 0:
        raise ValueError("source shelf probability schedule must be non-negative")
    if bool(getattr(args, "require_deployment_source", False)) and (
        str(getattr(args, "hierarchy_source_mode", "random")) == "random"
        or str(getattr(args, "hierarchy_source_levels", "random")) == "random"
    ):
        raise ValueError(
            "deployment-source training requires non-random block and level sources"
        )
    unfreeze_step = getattr(args, "backbone_unfreeze_step", None)
    if unfreeze_step is not None and int(unfreeze_step) < 0:
        raise ValueError("backbone unfreeze step must be non-negative")
    if (
        bool(getattr(args, "structured_legalizer_apply", False))
        and float(getattr(args, "legalizer_response_weight", 0.0)) <= 0.0
    ):
        raise ValueError(
            "legalizer action requires a positive legalizer response weight"
        )
    if (
        float(getattr(args, "legalizer_response_weight", 0.0)) > 0.0
        and getattr(args, "response_sidecar", None) is None
    ):
        raise ValueError("legalizer response loss requires a response sidecar")
    if bool(getattr(args, "structured_heads", False)):
        if bool(getattr(args, "membership_hierarchy_conditioning", False)):
            raise ValueError(
                "structured heads already own hierarchy coupling; "
                "disable membership hierarchy conditioning"
            )
        if bool(getattr(args, "standalone_legalizer_response", False)):
            raise ValueError(
                "structured heads already own legalizer response; "
                "disable the standalone legalizer response"
            )
    if (
        float(getattr(args, "endpoint_legalizer_target_weight", 0.0)) > 0.0
        and getattr(args, "response_sidecar", None) is None
    ):
        raise ValueError(
            "legalizer-target endpoint loss requires a response sidecar"
        )
    if getattr(args, "coarse_commit_fraction", None) is not None and (
        not bool(getattr(args, "structured_heads", False))
        or not bool(getattr(args, "structured_hierarchy_coupling", True))
    ):
        raise ValueError(
            "coarse commit requires same-call structured hierarchy coupling"
        )
    if bool(getattr(args, "relation_specific_attention", False)) and str(
        getattr(args, "typed_mode", "typed")
    ) == "dense":
        raise ValueError("relation-specific attention cannot use typed_mode=dense")
    if bool(getattr(args, "structural_relation_head", False)) and not (
        bool(getattr(args, "structured_heads", False))
        or bool(getattr(args, "hierarchy_token_bridge", False))
        or bool(getattr(args, "hierarchy_coarse_to_fine", False))
        or bool(getattr(args, "group_moment_conditioning", False))
    ):
        raise ValueError(
            "structural relation head requires a same-call hierarchy consumer"
        )
    if bool(getattr(args, "group_moment_conditioning", False)) and (
        not bool(getattr(args, "structural_relation_head", False))
        or float(getattr(args, "structural_relation_weight", 0.0)) <= 0.0
    ):
        raise ValueError(
            "group-moment conditioning requires structural relation supervision"
        )
    if not 0.0 <= float(getattr(args, "geometry_augment_probability", 1.0)) <= 1.0:
        raise ValueError("geometry augmentation probability must be in [0, 1]")
    if float(getattr(args, "backbone_lr_scale", 1.0)) <= 0.0:
        raise ValueError("backbone learning-rate scale must be positive")
    shape_modes = int(getattr(args, "shape_modes", 0))
    if shape_modes < 0:
        raise ValueError("shape mode count must be non-negative")
    if shape_modes > 0 and str(getattr(args, "architecture", "")) != "typed_hierarchy":
        raise ValueError("shape modes require typed_hierarchy")
    shape_width = getattr(args, "shape_mode_width", None)
    if shape_width is not None and int(shape_width) <= 0:
        raise ValueError("shape mode width must be positive")
    if float(getattr(args, "shape_mode_max_delta", 0.5)) <= 0.0:
        raise ValueError("shape mode max delta must be positive")
    if bool(getattr(args, "shape_mode_apply", False)) and shape_modes < 1:
        raise ValueError("shape mode application requires at least one mode")
    if str(getattr(args, "shape_mode_selection", "learned")) not in {
        "learned",
        "pareto",
    }:
        raise ValueError("shape mode selection must be learned or pareto")
    for name in (
        "shape_mode_loss_weight",
        "shape_mode_generator_weight",
        "shape_mode_diversity_weight",
        "shape_mode_usage_weight",
        "shape_mode_selection_weight",
        "shape_mode_diversity_margin",
        "shape_mode_center_scale",
    ):
        if float(getattr(args, name, 0.0)) < 0.0:
            raise ValueError(f"{name} must be non-negative")
    if float(getattr(args, "shape_mode_usage_temperature", 0.1)) <= 0.0:
        raise ValueError("shape mode usage temperature must be positive")
    if (
        float(getattr(args, "shape_mode_loss_weight", 0.0)) > 0.0
        and shape_modes < 2
    ):
        raise ValueError("shape-mode loss requires at least two modes")
    for name in ("structured_head_lr_scale", "legalizer_head_lr_scale"):
        if float(getattr(args, name, 1.0)) <= 0.0:
            raise ValueError(f"{name} must be positive")
    for name in ("structured_head_warmup_steps", "legalizer_head_warmup_steps"):
        if int(getattr(args, name, 0)) < 0:
            raise ValueError(f"{name} must be non-negative")
    if float(getattr(args, "grad_clip", 1.0)) < 0.0:
        raise ValueError("gradient clip norm must be non-negative")
    if float(getattr(args, "loss_balance_floor", 1.0e-6)) <= 0.0:
        raise ValueError("loss balance floor must be positive")
    if not 0.0 <= float(getattr(args, "ema_decay", 0.9995)) < 1.0:
        raise ValueError("EMA decay must be in [0, 1)")
    if int(getattr(args, "ema_warmup_steps", 1000)) < 0:
        raise ValueError("EMA warmup steps must be non-negative")
    if (
        str(getattr(args, "checkpoint_policy", "final")) == "best"
        and not getattr(args, "eval_steps", [])
    ):
        raise ValueError("best checkpoint selection requires evaluation steps")
    if int(getattr(args, "checkpoint_shortlist_size", 0)) < 0:
        raise ValueError("checkpoint shortlist size must be non-negative")
    if (
        str(getattr(args, "checkpoint_policy", "final")) == "legalized_best"
        and getattr(args, "checkpoint_selection_report", None) is None
    ):
        raise ValueError(
            "legalized_best policy requires --checkpoint-selection-report"
        )
    if float(getattr(args, "hierarchy_forced_head_weight", 0.0)) < 0.0:
        raise ValueError("forced hierarchy-head weight must be non-negative")
    if int(getattr(args, "endpoint_compatibility_iterations", 2)) < 0:
        raise ValueError("endpoint compatibility iterations must be non-negative")
    if int(getattr(args, "training_projector_iterations", 2)) < 0:
        raise ValueError("training projector iterations must be non-negative")
    if not 0.0 <= float(getattr(args, "training_projector_strength", 1.0)) <= 1.0:
        raise ValueError("training projector strength must be in [0, 1]")
    if not 0.0 <= float(
        getattr(args, "structured_legalizer_strength", 1.0)
    ) <= 1.0:
        raise ValueError("structured legalizer strength must be in [0, 1]")
    if float(
        getattr(args, "structured_legalizer_max_displacement", 0.1)
    ) < 0.0:
        raise ValueError("structured legalizer trust radius must be non-negative")
    if int(getattr(args, "projector_grid_size", 16)) < 1:
        raise ValueError("projector grid size must be positive")
    if int(getattr(args, "endpoint_refiner_trajectory_steps", 2)) < 0:
        raise ValueError("refiner trajectory steps must be non-negative")
    if int(getattr(args, "nfe_consistency_steps", 2)) <= 0:
        raise ValueError("NFE consistency steps must be positive")
    for name in (
        "endpoint_compatibility_weight",
        "endpoint_compatibility_displacement_weight",
        "endpoint_compatibility_gain_weight",
        "endpoint_refiner_energy_weight",
        "endpoint_refiner_trajectory_lr",
        "endpoint_refiner_trajectory_weight",
        "endpoint_legalized_area_weight",
        "endpoint_legalized_p2b_weight",
        "endpoint_legalizer_target_weight",
        "endpoint_mib_share_weight",
        "endpoint_pin_tail_weight",
        "endpoint_rho_trust_weight",
        "endpoint_rho_trust_radius",
        "nfe_consistency_weight",
        "nfe_consistency_target_weight",
        "nfe_consistency_self_weight",
        "official_cost_weight",
        "official_cost_final_weight",
        "official_cost_violation_temperature",
        "legalizer_response_weight",
        "legalizer_displacement_weight",
        "legalizer_rung_weight",
        "training_projector_displacement_weight",
        "structural_relation_weight",
        "relation_gate_weight",
    ):
        value = getattr(args, name, 0.0)
        if value is not None and float(value) < 0.0:
            raise ValueError(f"{name} must be non-negative")
    if not 0.0 < float(getattr(args, "endpoint_pin_tail_fraction", 0.2)) <= 1.0:
        raise ValueError("endpoint pin tail fraction must be in (0, 1]")
    if (
        float(getattr(args, "official_cost_weight", 0.0)) > 0.0
        or getattr(args, "official_cost_final_weight", None) is not None
    ) and float(
        getattr(args, "official_cost_violation_temperature", 0.1)
    ) <= 0.0:
        raise ValueError("official cost violation temperature must be positive")
    for name in (
        "official_cost_delay_steps",
        "official_cost_anneal_steps",
    ):
        if int(getattr(args, name, 0)) < 0:
            raise ValueError(f"{name} must be non-negative")


__all__ = [
    "validate_atom_metric_objective_schedule",
    "validate_training_scope",
]
