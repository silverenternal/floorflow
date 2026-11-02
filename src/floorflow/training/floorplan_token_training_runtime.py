"""Training-time contract for the canonical floorplan token branch."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

from floorflow.objectives.gradient_balancer import GradientNormBalancer
from floorflow.objectives.token_grammar import FloorplanTokenVocabulary


@dataclass(frozen=True)
class FloorplanTokenTrainingConfig:
    enabled: bool = False
    vocabulary_path: Path | None = None
    max_length: int = 512
    numeric_bin_size: int = 8
    geometry_token_mode: str = "reference"
    zero_oov: bool = True
    bucket_by_length: bool = True
    generation_eval_samples: int = 8
    generation_max_new_tokens: int = 512
    generation_budget_multiplier: float = 1.25
    generation_efficiency_budget: int = 64
    hard_validity_regression_gate: bool = False
    beam_width: int = 4
    space_syntax_rerank: bool = True
    hierarchy_weights: dict[str, float] = field(default_factory=dict)
    verifier_policy_weight: float = 0.0
    verifier_reward_baseline: float = 0.0
    verifier_reward_replay: dict[str, object] = field(default_factory=dict)
    conflict_interval: int = 5
    conflict_parameter_scope: str = "token_heads"
    balance_parameter_scope: str = "token_heads"
    conflict_projection: bool = False
    conflict_pairs: tuple[tuple[str, str], ...] = ()
    warmup_steps: int = 100
    structure_start_step: int = 100
    structure_ramp_steps: int = 100
    token_dropout_prob: float = 0.0
    prefix_corruption_prob: float = 0.0
    scheduled_sampling_start: int = 100
    scheduled_sampling_ramp: int = 100
    token_rollout_steps: int = 0
    weight: float = 0.0
    structure_weight: float = 0.0
    kind_weights: dict[str, float] = field(default_factory=dict)
    stage_weights: dict[str, float] = field(default_factory=dict)
    balance: dict[str, object] = field(default_factory=dict)


@dataclass
class FloorplanTokenTrainingRuntime:
    config: FloorplanTokenTrainingConfig
    vocabulary: FloorplanTokenVocabulary | None = None
    loss_balancer: GradientNormBalancer | None = None
    step: int = 0

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    def apply_model_config(self, model_config: dict[str, object]) -> None:
        if not self.enabled:
            model_config["enable_token_head"] = False
            return
        if self.vocabulary is None:
            raise RuntimeError("token vocabulary is not loaded")
        model_config["enable_token_head"] = True
        model_config["token_vocab_size"] = self.vocabulary.model_vocab_size
        model_config["token_max_length"] = self.config.max_length
        model_config.setdefault("token_head_layers", 2)

    def forward_kwargs(
        self,
        batch: dict,
        *,
        training: bool = True,
    ) -> dict[str, object]:
        if not self.enabled or "token_ids" not in batch:
            return {}
        token_ids = batch["token_ids"]
        token_mask = batch["token_mask"]
        input_ids = (
            self._scheduled_token_inputs(token_ids, token_mask)
            if training
            else token_ids
        )
        rollout_steps = (
            self.config.token_rollout_steps
            if training and self._schedule_scale() > 0.0
            else 0
        )
        return {
            "token_ids": batch["token_ids"],
            "token_mask": batch["token_mask"],
            "token_stage_ids": batch["token_stage_ids"],
            "token_input_ids": input_ids,
            "token_rollout_steps": rollout_steps,
        }

    def _schedule_scale(self) -> float:
        elapsed = self.step - self.config.scheduled_sampling_start + 1
        if elapsed <= 0:
            return 0.0
        return min(1.0, elapsed / max(1, self.config.scheduled_sampling_ramp))

    def _scheduled_token_inputs(
        self,
        token_ids: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        scale = self._schedule_scale()
        if scale <= 0.0 or self.vocabulary is None:
            return token_ids
        input_ids = token_ids.clone()
        positions = torch.arange(token_ids.shape[1], device=token_ids.device)
        modifiable = token_mask & (positions[None] >= min(5, token_ids.shape[1]))
        dropout_probability = float(self.config.token_dropout_prob) * scale
        if dropout_probability > 0.0:
            dropout = torch.rand_like(token_ids.float()) < dropout_probability
            replacements = torch.randint(
                0,
                self.vocabulary.model_vocab_size,
                token_ids.shape,
                device=token_ids.device,
            )
            input_ids = torch.where(
                dropout & modifiable,
                replacements,
                input_ids,
            )
        prefix_probability = float(self.config.prefix_corruption_prob) * scale
        if prefix_probability > 0.0:
            valid_count = modifiable.sum(dim=1)
            selected_rows = (
                torch.rand(token_ids.shape[0], device=token_ids.device)
                < prefix_probability
            ) & valid_count.gt(0)
            ordinal = torch.floor(
                torch.rand(token_ids.shape[0], device=token_ids.device)
                * valid_count.float()
            ).long().clamp_min(0)
            rank = modifiable.long().cumsum(dim=1) - 1
            selected_position = (
                modifiable & rank.eq(ordinal[:, None])
            ).float().argmax(dim=1)
            suffix = positions[None] >= selected_position[:, None]
            replacements = torch.randint(
                0,
                self.vocabulary.model_vocab_size,
                token_ids.shape,
                device=token_ids.device,
            )
            input_ids = torch.where(
                selected_rows[:, None] & suffix & modifiable,
                replacements,
                input_ids,
            )
        return input_ids.masked_fill(~token_mask, -1)

    def loss_kwargs(self) -> dict[str, object]:
        if not self.enabled:
            return {}
        return {
            "token_weight": self.scheduled_token_weight(),
            "token_kind_weights": dict(self.config.kind_weights),
            "token_stage_weights": dict(self.config.stage_weights),
            "token_structure_weight": self.scheduled_structure_weight(),
            "token_vocabulary": self.vocabulary,
            "loss_balancer": self.loss_balancer,
        }

    def scheduled_token_weight(self) -> float:
        scale = min(1.0, (self.step + 1) / max(1, self.config.warmup_steps))
        return float(self.config.weight) * scale

    def scheduled_structure_weight(self) -> float:
        elapsed = self.step - self.config.structure_start_step + 1
        if elapsed <= 0:
            return 0.0
        scale = min(
            1.0,
            elapsed / max(1, self.config.structure_ramp_steps),
        )
        return float(self.config.structure_weight) * scale

    def generation_budget(self, target_length: int) -> int:
        requested = max(
            1,
            int(float(target_length) * self.config.generation_budget_multiplier),
        )
        return min(
            self.config.max_length,
            self.config.generation_max_new_tokens,
            requested,
        )

    def checkpoint_payload(self) -> dict[str, object] | None:
        if not self.enabled or self.vocabulary is None:
            return None
        return {
            "protocol": "floorflow-canonical-token-training-v1",
            "vocabulary": self.vocabulary.to_json(),
            "vocabulary_sha256": self.vocabulary.sha256(),
            "max_length": self.config.max_length,
            "numeric_bin_size": self.config.numeric_bin_size,
            "geometry_token_mode": self.config.geometry_token_mode,
            "zero_oov": self.config.zero_oov,
            "bucket_by_length": self.config.bucket_by_length,
            "generation_eval_samples": self.config.generation_eval_samples,
            "generation_max_new_tokens": self.config.generation_max_new_tokens,
            "generation_budget_multiplier": self.config.generation_budget_multiplier,
            "generation_efficiency_budget": self.config.generation_efficiency_budget,
            "hard_validity_regression_gate": self.config.hard_validity_regression_gate,
            "beam_width": self.config.beam_width,
            "space_syntax_rerank": self.config.space_syntax_rerank,
            "hierarchy_weights": dict(self.config.hierarchy_weights),
            "verifier_policy_weight": self.config.verifier_policy_weight,
            "verifier_reward_baseline": self.config.verifier_reward_baseline,
            "verifier_reward_replay": dict(
                self.config.verifier_reward_replay
            ),
            "conflict_interval": self.config.conflict_interval,
            "conflict_parameter_scope": self.config.conflict_parameter_scope,
            "balance_parameter_scope": self.config.balance_parameter_scope,
            "conflict_projection": self.config.conflict_projection,
            "conflict_pairs": [list(pair) for pair in self.config.conflict_pairs],
            "warmup_steps": self.config.warmup_steps,
            "structure_start_step": self.config.structure_start_step,
            "structure_ramp_steps": self.config.structure_ramp_steps,
            "token_dropout_prob": self.config.token_dropout_prob,
            "prefix_corruption_prob": self.config.prefix_corruption_prob,
            "scheduled_sampling_start": self.config.scheduled_sampling_start,
            "scheduled_sampling_ramp": self.config.scheduled_sampling_ramp,
            "token_rollout_steps": self.config.token_rollout_steps,
            "weight": self.config.weight,
            "structure_weight": self.config.structure_weight,
            "kind_weights": dict(self.config.kind_weights),
            "stage_weights": dict(self.config.stage_weights),
            "loss_balancer": (
                self.loss_balancer.state_dict()
                if self.loss_balancer is not None
                else None
            ),
        }


def build_token_training_runtime(
    payload: dict[str, object] | None,
    *,
    root: Path,
    loss_weights: dict[str, float],
) -> FloorplanTokenTrainingRuntime:
    if not payload or not bool(payload.get("enabled", False)):
        return FloorplanTokenTrainingRuntime(
            FloorplanTokenTrainingConfig(enabled=False)
        )
    vocabulary_path = payload.get("vocabulary_path")
    if not vocabulary_path:
        raise ValueError("enabled token training requires vocabulary_path")
    path = Path(str(vocabulary_path))
    if not path.is_absolute():
        path = Path(root) / path
    config = FloorplanTokenTrainingConfig(
        enabled=True,
        vocabulary_path=path,
        max_length=int(payload.get("max_length", 512)),
        numeric_bin_size=int(payload.get("numeric_bin_size", 8)),
        geometry_token_mode=str(payload.get("geometry_token_mode", "reference")),
        zero_oov=bool(payload.get("zero_oov", True)),
        bucket_by_length=bool(payload.get("bucket_by_length", True)),
        generation_eval_samples=int(payload.get("generation_eval_samples", 8)),
        generation_max_new_tokens=int(payload.get("generation_max_new_tokens", 64)),
        generation_budget_multiplier=float(
            payload.get("generation_budget_multiplier", 1.25)
        ),
        generation_efficiency_budget=int(
            payload.get("generation_efficiency_budget", 64)
        ),
        hard_validity_regression_gate=bool(
            payload.get("hard_validity_regression_gate", False)
        ),
        beam_width=int(payload.get("beam_width", 4)),
        space_syntax_rerank=bool(payload.get("space_syntax_rerank", True)),
        hierarchy_weights={
            str(key): float(value)
            for key, value in dict(payload.get("hierarchy_weights", {})).items()
        },
        verifier_policy_weight=float(payload.get("verifier_policy_weight", 0.0)),
        verifier_reward_baseline=float(payload.get("verifier_reward_baseline", 0.0)),
        verifier_reward_replay=dict(
            payload.get("verifier_reward_replay") or {}
        ),
        conflict_interval=int(payload.get("conflict_interval", 5)),
        conflict_parameter_scope=str(
            payload.get("conflict_parameter_scope", "token_heads")
        ),
        balance_parameter_scope=str(
            payload.get("balance_parameter_scope", "token_heads")
        ),
        conflict_projection=bool(payload.get("conflict_projection", False)),
        conflict_pairs=tuple(
            (str(pair[0]), str(pair[1]))
            for pair in payload.get("conflict_pairs", ())
        ),
        warmup_steps=int(payload.get("warmup_steps", 100)),
        structure_start_step=int(payload.get("structure_start_step", 100)),
        structure_ramp_steps=int(payload.get("structure_ramp_steps", 100)),
        token_dropout_prob=float(payload.get("token_dropout_prob", 0.0)),
        prefix_corruption_prob=float(payload.get("prefix_corruption_prob", 0.0)),
        scheduled_sampling_start=int(payload.get("scheduled_sampling_start", 100)),
        scheduled_sampling_ramp=int(payload.get("scheduled_sampling_ramp", 100)),
        token_rollout_steps=int(payload.get("token_rollout_steps", 0)),
        weight=float(payload.get("weight", loss_weights.get("token", 0.1))),
        structure_weight=float(
            payload.get("structure_weight", loss_weights.get("token_structure", 0.0))
        ),
        kind_weights={
            str(key): float(value)
            for key, value in dict(payload.get("kind_weights", {})).items()
        },
        stage_weights={
            str(key): float(value)
            for key, value in dict(payload.get("stage_weights", {})).items()
        },
        balance=dict(payload.get("loss_balance", {})),
    )
    if config.max_length <= 0:
        raise ValueError("token max_length must be positive")
    if config.numeric_bin_size <= 0:
        raise ValueError("token numeric_bin_size must be positive")
    if config.geometry_token_mode not in {"reference", "quantized", "exact"}:
        raise ValueError("unknown token geometry mode")
    if config.generation_eval_samples < 0:
        raise ValueError("generation_eval_samples must be non-negative")
    if config.generation_max_new_tokens <= 0:
        raise ValueError("generation_max_new_tokens must be positive")
    if config.generation_budget_multiplier <= 0.0:
        raise ValueError("generation_budget_multiplier must be positive")
    if config.generation_efficiency_budget <= 0:
        raise ValueError("generation_efficiency_budget must be positive")
    if config.beam_width <= 0:
        raise ValueError("beam_width must be positive")
    if config.conflict_interval <= 0:
        raise ValueError("conflict_interval must be positive")
    if min(
        config.warmup_steps,
        config.structure_start_step,
        config.structure_ramp_steps,
    ) < 0:
        raise ValueError("token loss schedule values must be non-negative")
    if min(
        config.token_dropout_prob,
        config.prefix_corruption_prob,
    ) < 0.0 or max(
        config.token_dropout_prob,
        config.prefix_corruption_prob,
    ) > 1.0:
        raise ValueError("token corruption probabilities must be in [0,1]")
    if config.token_rollout_steps < 0:
        raise ValueError("token_rollout_steps must be non-negative")
    if config.weight <= 0.0:
        raise ValueError("enabled token training requires positive weight")
    vocabulary = FloorplanTokenVocabulary.read(path)
    balancer = None
    balance = config.balance
    if bool(balance.get("enabled", False)):
        terms = (
            "polygon",
            "identity",
            "count",
            "wall",
            "opening",
            "adjacency",
            "token",
            "token_structure",
        )
        balancer = GradientNormBalancer(
            base_weights={name: 1.0 for name in terms},
            reference=str(balance.get("reference", "polygon")),
            momentum=float(balance.get("momentum", 0.95)),
            minimum_scale=float(balance.get("minimum_scale", 0.25)),
            maximum_scale=float(balance.get("maximum_scale", 1.5)),
            update_interval=int(balance.get("update_interval", 100)),
        )
    return FloorplanTokenTrainingRuntime(
        config=config,
        vocabulary=vocabulary,
        loss_balancer=balancer,
    )


__all__ = [
    "FloorplanTokenTrainingConfig",
    "FloorplanTokenTrainingRuntime",
    "build_token_training_runtime",
]
