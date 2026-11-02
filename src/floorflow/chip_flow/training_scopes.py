"""Parameter-isolated training stages for typed chip channels and fusion."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from torch import nn

from floorflow.chip_flow.objective_schedules import (
    learning_rate,
    stage_learning_rate,
)

CHANNEL_PREFIXES: dict[str, tuple[str, ...]] = {
    "block": ("typed.block_input.", "typed.graph_layers."),
    "local": ("typed.local_geometry_encoder.",),
    "global": (
        "typed.global_spec_conditioner.",
        "typed.complexity.",
    ),
    "structural": ("typed.structural_type",),
    "constraint": ("typed.constraint_type",),
    "port": ("typed.port_input.", "typed.port_type"),
    "semantic": (
        "typed.node_semantic_encoder.",
        "typed.port_semantic_encoder.",
    ),
    "expert": ("typed.degree_expert.",),
    "bridge": (
        "typed.global_to_structure.",
        "typed.constraint_to_port.",
    ),
}

FUSION_PREFIXES = (
    "typed.cross_attention.",
    "typed.norm.",
    "structured_heads.atom_aggregator.",
    "time.",
)

ATOM_SCOPE_PREFIXES: dict[str, tuple[str, ...]] = {
    "area": (
        "typed.global_spec_conditioner.",
        "typed.complexity.",
        "typed.global_to_structure.",
        "coordinate_gate.",
        "structured_heads.factorized_block.",
        "shape_mode_head.",
        "structured_heads.hierarchy_coupling.",
        "global_bbox_consensus.",
        "global_head.",
        "hierarchy_summary.",
        "hierarchy_to_latent.",
        "projector_residual_head.",
        "observed_hierarchy_coupler.global_adapter.",
    ),
    "b2b": (
        "typed.block_input.",
        "typed.graph_layers.",
        "typed.local_geometry_encoder.",
        "typed.degree_expert.",
        "typed.structural_type",
        "relation_head.",
        "group_moment_coupler.",
        "dynamic_relay.",
        "dynamic_to_latent.",
        "structural_head.",
        "conditioners.hyperedge.",
        "observed_hierarchy_coupler.structural_adapter.",
    ),
    "port_route": (
        "typed.port_input.",
        "typed.port_type",
        "typed.constraint_to_port.",
        "port_consensus.",
        "port_head.",
        "structured_heads.dual_semantic_ports.",
        "spatial_occupancy.",
        "observed_hierarchy_coupler.port_adapter.",
    ),
    "boundary": (
        "typed.constraint_type",
        "boundary_anchor.",
    ),
    "tail": (
        "complexity_slot.",
        "conditioners.legalizer_head.",
        "structured_heads.legalizer_response.",
        "projector_residual_head.",
    ),
}
ATOM_GROUPS = frozenset(ATOM_SCOPE_PREFIXES)


@dataclass(frozen=True)
class TrainingScope:
    groups: tuple[str, ...]

    @property
    def name(self) -> str:
        return "+".join(self.groups)


@dataclass(frozen=True)
class TrainingScopeTransition:
    start: int
    scope: TrainingScope


def parse_training_scope(value: str) -> TrainingScope:
    """Parse a composable scope such as ``channel:local+downstream``."""

    raw = tuple(part.strip() for part in value.split("+") if part.strip())
    if not raw:
        raise ValueError("training scope must not be empty")
    aliases = {"channel:" + name: name for name in CHANNEL_PREFIXES}
    aliases["channels"] = "channels"
    aliases["fusion"] = "fusion"
    aliases["downstream"] = "downstream"
    aliases["typed"] = "typed"
    aliases["all"] = "all"
    for name in ATOM_SCOPE_PREFIXES:
        aliases[name] = name
    groups = []
    for token in raw:
        group = aliases.get(token, token if token in aliases.values() else None)
        if group is None:
            raise ValueError(f"unknown training scope group: {token}")
        if group not in groups:
            groups.append(group)
    if "all" in groups and len(groups) != 1:
        raise ValueError("all cannot be combined with other training scope groups")
    return TrainingScope(tuple(groups))


def validate_atom_isolation_schedule(
    schedule: tuple[TrainingScopeTransition, ...],
) -> dict[str, object]:
    """Require one owned atom per stage and one final fusion-only stage."""

    if not schedule:
        raise ValueError("atom isolation requires a training scope schedule")
    atom_counts = {name: 0 for name in ATOM_GROUPS}
    for transition in schedule:
        groups = set(transition.scope.groups)
        atoms = groups & ATOM_GROUPS
        unsupported = groups - ATOM_GROUPS - {"fusion", "downstream"}
        if unsupported:
            raise ValueError(
                "atom isolation forbids shared scope groups: "
                f"{sorted(unsupported)}"
            )
        if len(atoms) > 1:
            raise ValueError(
                "atom isolation requires one atom per stage: "
                f"{sorted(atoms)}"
            )
        if "downstream" in groups and "fusion" not in groups:
            raise ValueError(
                "downstream atom stages must be coupled with fusion"
            )
        for atom in atoms:
            atom_counts[atom] += 1
    missing = sorted(name for name, count in atom_counts.items() if count != 1)
    if missing:
        raise ValueError(
            "atom isolation requires each atom exactly once: "
            f"{missing}"
        )
    final_groups = set(schedule[-1].scope.groups)
    if final_groups != {"fusion"}:
        raise ValueError(
            "atom isolation must end with fusion only"
        )
    return {
        "atoms": atom_counts,
        "stages": len(schedule),
        "final_scope": schedule[-1].scope.name,
    }


def _prefix_group(name: str, group: str) -> bool:
    if group == "downstream":
        return not name.startswith("typed.")
    if group == "typed":
        return name.startswith("typed.")
    if group == "fusion" or group in ATOM_SCOPE_PREFIXES:
        return group in scope_owners(name)
    if group == "channels":
        return any(name.startswith(prefix) for values in CHANNEL_PREFIXES.values() for prefix in values)
    prefixes = CHANNEL_PREFIXES.get(group, ())
    return name.startswith(prefixes)


def scope_owners(name: str) -> tuple[str, ...]:
    """Return atom/fusion groups that explicitly own a parameter name."""

    owners = tuple(
        group
        for group, prefixes in ATOM_SCOPE_PREFIXES.items()
        if name.startswith(prefixes)
    )
    if name.startswith(FUSION_PREFIXES):
        owners = (*owners, "fusion")
    return owners


def apply_training_scope(model: nn.Module, scope: TrainingScope) -> dict[str, object]:
    """Freeze everything outside the selected scope and report its exact extent."""

    named = tuple(model.named_parameters())
    if "all" in scope.groups:
        for _, parameter in named:
            parameter.requires_grad_(True)
    else:
        for name, parameter in named:
            parameter.requires_grad_(
                any(_prefix_group(name, group) for group in scope.groups)
            )
    selected = [
        (name, parameter)
        for name, parameter in named
        if parameter.requires_grad
    ]
    if not selected:
        raise ValueError(f"training scope selects no parameters: {scope.name}")
    return {
        "scope": scope.name,
        "groups": scope.groups,
        "trainable_tensors": len(selected),
        "trainable_parameters": sum(parameter.numel() for _, parameter in selected),
        "trainable_prefixes": tuple(
            sorted({".".join(name.split(".")[:2]) for name, _ in selected})
        ),
    }


def typed_channel_encoding_requires_grad(model: nn.Module) -> bool:
    """Return whether any pre-fusion typed channel parameter needs gradients."""

    return any(
        parameter.requires_grad
        and not name.startswith(FUSION_PREFIXES)
        for name, parameter in model.named_parameters()
        if name.startswith("typed.")
    )


def attach_typed_channel_cache(model: nn.Module, args: object) -> str | None:
    root = getattr(args, "typed_channel_cache_root", None)
    if root is None:
        return None
    model.typed_channel_cache_root = str(root)
    model.typed_channel_cache_split = str(
        getattr(args, "typed_channel_cache_split", "train")
    )
    return model.typed_channel_cache_root


def parse_scope_schedule(value: str | None) -> tuple[TrainingScopeTransition, ...]:
    """Parse ``start:scope`` transitions separated by commas."""

    if value is None:
        return ()
    transitions = []
    previous = -1
    for item in value.split(","):
        if not item.strip():
            continue
        try:
            start_text, scope_text = item.split(":", 1)
            start = int(start_text)
        except ValueError as error:
            raise ValueError("scope schedule entries must use start:scope") from error
        if start < 0 or start <= previous:
            raise ValueError("scope schedule starts must be non-negative and increasing")
        transitions.append(
            TrainingScopeTransition(start=start, scope=parse_training_scope(scope_text))
        )
        previous = start
    return tuple(transitions)


def scope_at_step(
    schedule: tuple[TrainingScopeTransition, ...],
    step: int,
) -> TrainingScope | None:
    """Return the latest scheduled scope at a training step."""

    selected = None
    for transition in schedule:
        if int(step) < transition.start:
            break
        selected = transition.scope
    return selected


class ChannelStageController:
    """Apply one static scope or stepwise channel-to-fusion transitions."""

    def __init__(
        self,
        initial: TrainingScope,
        schedule: tuple[TrainingScopeTransition, ...],
    ) -> None:
        self.initial = initial
        self.schedule = schedule
        self.current = initial
        self.stats: dict[str, object] | None = None
        self.history: list[dict[str, object]] = []

    @classmethod
    def from_args(cls, args: object) -> ChannelStageController:
        schedule_text = getattr(args, "training_scope_schedule", None)
        initial = parse_training_scope(
            str(getattr(args, "training_scope", "all"))
        )
        schedule = parse_scope_schedule(
            None if schedule_text is None else str(schedule_text)
        )
        if schedule:
            if schedule[0].start != 0:
                raise ValueError("training scope schedule must start at step zero")
            initial = schedule[0].scope
        return cls(initial=initial, schedule=schedule)

    @property
    def requires_all_optimizer_parameters(self) -> bool:
        return bool(self.schedule)

    def apply(self, model: nn.Module, *, step: int = 0) -> bool:
        self.stats = apply_training_scope(model, self.current)
        self.stats["step"] = int(step)
        self.history.append(self.stats)
        return True

    def apply_initial(self, model: nn.Module) -> None:
        """Preserve explicit freeze flags when the default all scope is unused."""

        if not self.schedule and self.current.name == "all":
            selected = [
                parameter for parameter in model.parameters() if parameter.requires_grad
            ]
            self.stats = {
                "scope": "all",
                "mode": "preserve_existing_flags",
                "trainable_tensors": len(selected),
                "trainable_parameters": sum(
                    parameter.numel() for parameter in selected
                ),
                "step": 0,
            }
            self.history.append(self.stats)
            return
        self.apply(model, step=0)

    def update(self, model: nn.Module, step: int) -> bool:
        selected = scope_at_step(self.schedule, step)
        if selected is None or selected.name == self.current.name:
            return False
        self.current = selected
        return self.apply(model, step=step)

    def window_at_step(self, step: int, total_steps: int) -> tuple[int, int]:
        """Return the active stage interval for stage-local schedules."""

        if not self.schedule:
            return 0, max(1, int(total_steps))
        selected = 0
        for index, transition in enumerate(self.schedule):
            if int(step) < transition.start:
                break
            selected = index
        start = self.schedule[selected].start
        end = (
            self.schedule[selected + 1].start
            if selected + 1 < len(self.schedule)
            else max(start + 1, int(total_steps))
        )
        return start, end

    def learning_rate(self, step: int, args: argparse.Namespace) -> float:
        """Apply global LR or restart it for each isolated stage."""

        if not bool(getattr(args, "stage_lr_restart", False)):
            return learning_rate(step, args)
        start, end = self.window_at_step(step, args.steps)
        return stage_learning_rate(
            args.lr,
            step - start,
            end - start,
            args.stage_lr_warmup_steps,
        )


__all__ = [
    "ATOM_GROUPS",
    "ATOM_SCOPE_PREFIXES",
    "CHANNEL_PREFIXES",
    "FUSION_PREFIXES",
    "ChannelStageController",
    "TrainingScope",
    "TrainingScopeTransition",
    "apply_training_scope",
    "attach_typed_channel_cache",
    "parse_scope_schedule",
    "parse_training_scope",
    "scope_at_step",
    "scope_owners",
    "typed_channel_encoding_requires_grad",
    "validate_atom_isolation_schedule",
]
