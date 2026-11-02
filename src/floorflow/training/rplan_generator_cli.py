"""Command-line contract for the unified RPLAN generator."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[3]
    / "configs/rplan_generation_sota_v1.json"
)


@dataclass(frozen=True)
class TrainConfig:
    config_path: Path
    output_dir: Path
    init_checkpoint: Path | None = None
    freeze_non_stage: bool = False
    epochs: int | None = None
    directory: str = "train"
    validation_directory: str = "val"
    train_cases: int | None = None
    validation_cases: int | None = None
    train_offset: int = 0
    validation_offset: int = 0
    batch_size: int | None = None
    learning_rate: float | None = None
    workers: int | None = None
    device: str | None = None
    seed: int | None = None
    contract_smoke: bool = False
    disable_identity_fusion: bool = False
    joint_mask_match_weight: float | None = None
    enable_joint_room_assignment: bool = False
    query_separation_weight: float | None = None
    class_balance_power: float | None = None
    class_aware_separation: bool = False
    supervised_separation: bool = False
    query_quality_weight: float | None = None
    enable_query_quality: bool = False
    point_validity_weight: float | None = None
    enable_point_validity: bool = False
    partition_consistency_weight: float | None = None
    enable_partition_consistent_queries: bool = False
    use_boundary_edge_graph: bool = False
    compact_encoder: bool = False
    enable_program_set: bool = False
    room_program_weight: float | None = None
    disable_boundary_edge_graph: bool = False
    disable_point_validity: bool = False
    disable_program_set: bool = False
    enable_query_graph_refiner: bool = False
    disable_query_graph_refiner: bool = False
    separate_identity_heads: bool = False
    pretrained_backbone: bool = False
    readout: str = "polygon"
    encoder_architecture: str | None = None
    enable_token_head: bool = False
    token_vocabulary: Path | None = None
    token_max_length: int | None = None
    token_weight: float | None = None
    token_warmup_steps: int | None = None
    token_structure_start_step: int | None = None
    token_structure_ramp_steps: int | None = None
    token_conflict_projection: bool = False
    token_scheduled_sampling_start: int | None = None
    token_dropout_prob: float | None = None
    token_prefix_corruption_prob: float | None = None
    token_rollout_steps: int | None = None
    token_generation_eval_samples: int | None = None
    token_beam_width: int | None = None
    precision: str | None = None
    compile_model: bool = False
    compile_mode: str = "default"
    channels_last: bool = False
    tf32: bool = True
    disable_triton_kernels: bool = False
    prefetch_factor: int | None = None
    verifier_reward_refresh_interval: int | None = None
    verifier_reward_max_age_steps: int | None = None
    verifier_reward_replay: Path | None = None
    write_verifier_reward_replay: Path | None = None
    profile_dir: Path | None = None


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--freeze-non-stage", action="store_true")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--train-cases", type=int)
    parser.add_argument("--validation-cases", type=int)
    parser.add_argument("--train-offset", type=int, default=0)
    parser.add_argument("--validation-offset", type=int, default=0)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--contract-smoke", action="store_true")
    parser.add_argument("--disable-identity-fusion", action="store_true")
    parser.add_argument("--joint-mask-match-weight", type=float)
    parser.add_argument("--enable-joint-room-assignment", action="store_true")
    parser.add_argument("--query-separation-weight", type=float)
    parser.add_argument("--class-balance-power", type=float)
    parser.add_argument("--class-aware-separation", action="store_true")
    parser.add_argument("--supervised-separation", action="store_true")
    parser.add_argument("--query-quality-weight", type=float)
    parser.add_argument("--enable-query-quality", action="store_true")
    parser.add_argument("--point-validity-weight", type=float)
    parser.add_argument("--enable-point-validity", action="store_true")
    parser.add_argument("--partition-consistency-weight", type=float)
    parser.add_argument(
        "--enable-partition-consistent-queries",
        action="store_true",
    )
    parser.add_argument("--use-boundary-edge-graph", action="store_true")
    parser.add_argument("--compact-encoder", action="store_true")
    parser.add_argument("--enable-program-set", action="store_true")
    parser.add_argument("--room-program-weight", type=float)
    parser.add_argument("--disable-boundary-edge-graph", action="store_true")
    parser.add_argument("--disable-point-validity", action="store_true")
    parser.add_argument("--disable-program-set", action="store_true")
    parser.add_argument("--enable-query-graph-refiner", action="store_true")
    parser.add_argument("--disable-query-graph-refiner", action="store_true")
    parser.add_argument("--separate-identity-heads", action="store_true")
    parser.add_argument("--pretrained-backbone", action="store_true")
    parser.add_argument(
        "--readout",
        choices=("polygon", "mask_partition"),
        default="polygon",
    )
    parser.add_argument(
        "--encoder-architecture",
        choices=("resnet18", "compact"),
    )
    parser.add_argument("--enable-token-head", action="store_true")
    parser.add_argument("--token-vocabulary", type=Path)
    parser.add_argument("--token-max-length", type=int)
    parser.add_argument("--token-weight", type=float)
    parser.add_argument("--token-warmup-steps", type=int)
    parser.add_argument("--token-structure-start-step", type=int)
    parser.add_argument("--token-structure-ramp-steps", type=int)
    parser.add_argument("--token-conflict-projection", action="store_true")
    parser.add_argument("--token-scheduled-sampling-start", type=int)
    parser.add_argument("--token-dropout-prob", type=float)
    parser.add_argument("--token-prefix-corruption-prob", type=float)
    parser.add_argument("--token-rollout-steps", type=int)
    parser.add_argument("--token-generation-eval-samples", type=int)
    parser.add_argument("--token-beam-width", type=int)
    parser.add_argument(
        "--precision",
        choices=("fp32", "fp16", "bf16"),
    )
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="default",
    )
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--disable-tf32", action="store_true")
    parser.add_argument("--disable-triton-kernels", action="store_true")
    parser.add_argument("--prefetch-factor", type=int)
    parser.add_argument("--verifier-reward-refresh-interval", type=int)
    parser.add_argument("--verifier-reward-max-age-steps", type=int)
    parser.add_argument("--verifier-reward-replay", type=Path)
    parser.add_argument("--write-verifier-reward-replay", type=Path)
    parser.add_argument("--profile-dir", type=Path)
    args = parser.parse_args()
    return TrainConfig(
        config_path=args.config,
        output_dir=args.output_dir,
        init_checkpoint=args.init_checkpoint,
        freeze_non_stage=args.freeze_non_stage,
        epochs=args.epochs,
        train_cases=args.train_cases,
        validation_cases=args.validation_cases,
        train_offset=args.train_offset,
        validation_offset=args.validation_offset,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        workers=args.workers,
        device=args.device,
        seed=args.seed,
        contract_smoke=args.contract_smoke,
        disable_identity_fusion=args.disable_identity_fusion,
        joint_mask_match_weight=args.joint_mask_match_weight,
        enable_joint_room_assignment=args.enable_joint_room_assignment,
        query_separation_weight=args.query_separation_weight,
        class_balance_power=args.class_balance_power,
        class_aware_separation=args.class_aware_separation,
        supervised_separation=args.supervised_separation,
        query_quality_weight=args.query_quality_weight,
        enable_query_quality=args.enable_query_quality,
        point_validity_weight=args.point_validity_weight,
        enable_point_validity=args.enable_point_validity,
        partition_consistency_weight=args.partition_consistency_weight,
        enable_partition_consistent_queries=(
            args.enable_partition_consistent_queries
        ),
        use_boundary_edge_graph=args.use_boundary_edge_graph,
        compact_encoder=args.compact_encoder,
        enable_program_set=args.enable_program_set,
        room_program_weight=args.room_program_weight,
        disable_boundary_edge_graph=args.disable_boundary_edge_graph,
        disable_point_validity=args.disable_point_validity,
        disable_program_set=args.disable_program_set,
        enable_query_graph_refiner=args.enable_query_graph_refiner,
        disable_query_graph_refiner=args.disable_query_graph_refiner,
        separate_identity_heads=args.separate_identity_heads,
        pretrained_backbone=args.pretrained_backbone,
        readout=args.readout,
        encoder_architecture=args.encoder_architecture,
        enable_token_head=args.enable_token_head,
        token_vocabulary=args.token_vocabulary,
        token_max_length=args.token_max_length,
        token_weight=args.token_weight,
        token_warmup_steps=args.token_warmup_steps,
        token_structure_start_step=args.token_structure_start_step,
        token_structure_ramp_steps=args.token_structure_ramp_steps,
        token_conflict_projection=args.token_conflict_projection,
        token_scheduled_sampling_start=args.token_scheduled_sampling_start,
        token_dropout_prob=args.token_dropout_prob,
        token_prefix_corruption_prob=args.token_prefix_corruption_prob,
        token_rollout_steps=args.token_rollout_steps,
        token_generation_eval_samples=args.token_generation_eval_samples,
        token_beam_width=args.token_beam_width,
        precision=args.precision,
        compile_model=args.compile_model,
        compile_mode=args.compile_mode,
        channels_last=args.channels_last,
        tf32=not args.disable_tf32,
        disable_triton_kernels=args.disable_triton_kernels,
        prefetch_factor=args.prefetch_factor,
        verifier_reward_refresh_interval=(
            args.verifier_reward_refresh_interval
        ),
        verifier_reward_max_age_steps=args.verifier_reward_max_age_steps,
        verifier_reward_replay=args.verifier_reward_replay,
        write_verifier_reward_replay=args.write_verifier_reward_replay,
        profile_dir=args.profile_dir,
    )


__all__ = ["DEFAULT_CONFIG", "TrainConfig", "parse_args"]
