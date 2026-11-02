"""Train and audit the unified RPLAN boundary-to-CAD generator."""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from floorflow.data.length_bucketing import LengthBucketBatchSampler
from floorflow.data.rplan_boundary_adapter import (
    RPlanBoundaryDataset,
    RPlanTensorLimits,
    collate_rplan_items,
    load_rplan_rows,
    rplan_taxonomy,
)
from floorflow.data.rplan_model_limits import validate_rplan_model_limits
from floorflow.evaluation.fevp_metric_contracts import FEVPTolerances
from floorflow.model.unified_floorplan_generator import (
    UnifiedFloorplanGenerator,
    UnifiedGeneratorConfig,
)
from floorflow.objectives.token_soft_graph import configure_token_graph_catalogue
from floorflow.training.floorplan_token_training_runtime import (
    build_token_training_runtime,
)
from floorflow.training.gpu_profiler import profile_training_epoch
from floorflow.training.rplan_full_cad_epoch import run_epoch
from floorflow.training.rplan_generator_cli import parse_args
from floorflow.training.rplan_runtime_setup import (
    build_acceleration_runtime,
    build_verifier_replay,
)
from floorflow.training.rplan_selection import rplan_selection_score
from floorflow.training.token_optimizer import build_floorflow_optimizer

ROOT = Path(__file__).resolve().parents[1]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main() -> None:
    cli = parse_args()
    config_payload = json.loads(
        cli.config_path.read_text(encoding="utf-8")
    )
    training = config_payload["training"]
    seed = int(cli.seed if cli.seed is not None else training["seed"])
    set_seed(seed)
    device_name = cli.device or str(training["device"])
    if device_name.startswith("cuda") and device_name != "cuda:0":
        raise ValueError("FloorFlow GPU training is restricted to cuda:0")
    device = torch.device(device_name)
    runtime = build_acceleration_runtime(cli, config_payload, device)
    taxonomy = rplan_taxonomy()
    tensor_config = config_payload["tensors"]
    limits = RPlanTensorLimits(
        **{key: int(value) for key, value in tensor_config.items()}
    )
    token_payload = dict(config_payload.get("tokenization") or {})
    if cli.enable_token_head:
        token_payload["enabled"] = True
    if cli.token_vocabulary is not None:
        token_payload["enabled"] = True
        token_payload["vocabulary_path"] = str(cli.token_vocabulary)
    if cli.token_max_length is not None:
        token_payload["max_length"] = int(cli.token_max_length)
    if cli.token_weight is not None:
        token_payload["weight"] = float(cli.token_weight)
    if cli.token_warmup_steps is not None:
        token_payload["warmup_steps"] = int(cli.token_warmup_steps)
    if cli.token_structure_start_step is not None:
        token_payload["structure_start_step"] = int(
            cli.token_structure_start_step
        )
    if cli.token_structure_ramp_steps is not None:
        token_payload["structure_ramp_steps"] = int(
            cli.token_structure_ramp_steps
        )
    if cli.token_conflict_projection:
        token_payload["conflict_projection"] = True
    if cli.token_scheduled_sampling_start is not None:
        token_payload["scheduled_sampling_start"] = int(
            cli.token_scheduled_sampling_start
        )
    if cli.token_dropout_prob is not None:
        token_payload["token_dropout_prob"] = float(cli.token_dropout_prob)
    if cli.token_prefix_corruption_prob is not None:
        token_payload["prefix_corruption_prob"] = float(
            cli.token_prefix_corruption_prob
        )
    if cli.token_rollout_steps is not None:
        token_payload["token_rollout_steps"] = int(cli.token_rollout_steps)
    if cli.token_generation_eval_samples is not None:
        token_payload["generation_eval_samples"] = int(
            cli.token_generation_eval_samples
        )
    if cli.token_beam_width is not None:
        token_payload["beam_width"] = int(cli.token_beam_width)
    token_runtime = build_token_training_runtime(
        token_payload,
        root=ROOT,
        loss_weights={
            str(key): float(value)
            for key, value in config_payload.get("loss_weights", {}).items()
            if isinstance(value, (int, float))
        },
    )
    train_cases = int(
        cli.train_cases
        if cli.train_cases is not None
        else training["train_cases"]
    )
    validation_cases = int(
        cli.validation_cases
        if cli.validation_cases is not None
        else training["validation_cases"]
    )
    if cli.contract_smoke:
        train_cases = min(train_cases, int(training["batch_size"]))
        validation_cases = min(validation_cases, 1)
    train_path = ROOT / config_payload["dataset"]["train"]
    val_path = ROOT / config_payload["dataset"]["validation"]
    train_dataset = RPlanBoundaryDataset(
        train_path,
        taxonomy,
        limits=limits,
        rows=load_rplan_rows(
            train_path,
            offset=cli.train_offset,
            limit=train_cases,
        ),
        token_vocabulary=token_runtime.vocabulary,
        token_max_length=token_runtime.config.max_length,
        token_numeric_bin_size=token_runtime.config.numeric_bin_size,
        token_geometry_mode=token_runtime.config.geometry_token_mode,
    )
    val_dataset = RPlanBoundaryDataset(
        val_path,
        taxonomy,
        limits=limits,
        rows=load_rplan_rows(
            val_path,
            offset=cli.validation_offset,
            limit=validation_cases,
        ),
        token_vocabulary=token_runtime.vocabulary,
        token_max_length=token_runtime.config.max_length,
        token_numeric_bin_size=token_runtime.config.numeric_bin_size,
        token_geometry_mode=token_runtime.config.geometry_token_mode,
    )
    batch_size = int(
        cli.batch_size
        if cli.batch_size is not None
        else training["batch_size"]
    )
    workers = int(
        cli.workers
        if cli.workers is not None
        else training["workers"]
    )
    bucket_sampler = None
    if token_runtime.enabled and token_runtime.config.bucket_by_length:
        bucket_sampler = LengthBucketBatchSampler(
            [train_dataset.token_length(index) for index in range(len(train_dataset))],
            batch_size,
            shuffle=True,
            seed=seed,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=bucket_sampler,
            collate_fn=collate_rplan_items,
            **runtime.dataloader_options(device, workers),
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_rplan_items,
            **runtime.dataloader_options(device, workers),
        )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_rplan_items,
        **runtime.dataloader_options(device, workers),
    )
    model_config = dict(config_payload["model"])
    if cli.disable_identity_fusion:
        model_config["use_mask_anchors"] = False
        model_config["share_identity_heads"] = False
    if cli.enable_query_quality:
        model_config["enable_query_quality"] = True
    if cli.enable_point_validity:
        model_config["enable_point_validity"] = True
    if cli.enable_partition_consistent_queries:
        model_config["enable_partition_consistent_queries"] = True
    if cli.use_boundary_edge_graph:
        model_config["use_boundary_edge_graph"] = True
    if cli.compact_encoder:
        model_config["encoder_architecture"] = "compact"
    if cli.enable_program_set:
        model_config["enable_program_set"] = True
    if cli.disable_boundary_edge_graph:
        model_config["use_boundary_edge_graph"] = False
    if cli.disable_point_validity:
        model_config["enable_point_validity"] = False
    if cli.disable_program_set:
        model_config["enable_program_set"] = False
    if cli.enable_query_graph_refiner:
        model_config["enable_query_graph_refiner"] = True
    if cli.disable_query_graph_refiner:
        model_config["enable_query_graph_refiner"] = False
    if cli.separate_identity_heads:
        model_config["share_identity_heads"] = False
    if cli.pretrained_backbone:
        model_config["pretrained_backbone"] = True
    if cli.encoder_architecture is not None:
        model_config["encoder_architecture"] = cli.encoder_architecture
    loss_weights = dict(config_payload["loss_weights"])
    if token_runtime.enabled:
        loss_weights["token"] = token_runtime.config.weight
        loss_weights["token_structure"] = token_runtime.config.structure_weight
    token_runtime.apply_model_config(model_config)
    if not bool(model_config.get("enable_program_set", False)):
        loss_weights["room_program"] = 0.0
    if not bool(model_config.get("enable_point_validity", False)):
        loss_weights["point_validity"] = 0.0
    validate_rplan_model_limits(limits, model_config)
    model = UnifiedFloorplanGenerator(
        taxonomy,
        UnifiedGeneratorConfig(**model_config),
    ).to(device)
    if token_runtime.enabled:
        configure_token_graph_catalogue(
            model,
            token_runtime.vocabulary,
            taxonomy,
        )
    init_status = None
    initial_optimizer_state = None
    initial_grad_scaler_state = None
    if cli.init_checkpoint is not None:
        payload = torch.load(
            cli.init_checkpoint,
            map_location=device,
            weights_only=False,
        )
        checkpoint_token = payload.get("token_training")
        if token_runtime.enabled and checkpoint_token is not None:
            expected_sha = token_runtime.vocabulary.sha256()
            if checkpoint_token.get("vocabulary_sha256") != expected_sha:
                raise ValueError("init checkpoint token vocabulary mismatch")
            expected_contract = {
                "max_length": token_runtime.config.max_length,
                "numeric_bin_size": token_runtime.config.numeric_bin_size,
                "geometry_token_mode": token_runtime.config.geometry_token_mode,
            }
            for key, expected in expected_contract.items():
                if checkpoint_token.get(key) != expected:
                    raise ValueError(
                        f"init checkpoint token contract mismatch: {key}"
                    )
            balancer_state = checkpoint_token.get("loss_balancer")
            if token_runtime.loss_balancer is not None and balancer_state:
                token_runtime.loss_balancer.load_state_dict(balancer_state)
            token_runtime.step = int(payload.get("global_step", 0))
        initial_optimizer_state = payload.get("optimizer")
        initial_grad_scaler_state = payload.get("grad_scaler")
        load_result = model.load_state_dict(payload["model"], strict=False)
        token_prefixes = (
            "token_head.",
            "query_pool.",
            "token_memory.",
            "token_refiner.",
            "room_instance_head.",
        )
        disallowed_missing = [
            key
            for key in load_result.missing_keys
            if "stage_conditioners." not in key
            and not (token_runtime.enabled and key.startswith(token_prefixes))
        ]
        if disallowed_missing or load_result.unexpected_keys:
            raise ValueError(
                "init checkpoint mismatch: "
                f"missing={disallowed_missing}; "
                f"unexpected={load_result.unexpected_keys}"
            )
        init_status = {
            "checkpoint": str(cli.init_checkpoint),
            "missing_stage_keys": len(load_result.missing_keys),
            "unexpected_keys": len(load_result.unexpected_keys),
        }
    model = runtime.prepare_model(model)
    if cli.freeze_non_stage:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = "stage_conditioners." in name
    optimizer = build_floorflow_optimizer(
        model,
        learning_rate=float(
            cli.learning_rate
            if cli.learning_rate is not None
            else training["learning_rate"]
        ),
        weight_decay=float(training["weight_decay"]),
        token_learning_rate_multiplier=float(
            training.get("token_learning_rate_multiplier", 1.0)
        ),
    )
    if initial_optimizer_state is not None:
        optimizer.load_state_dict(initial_optimizer_state)
    tolerances = FEVPTolerances()
    grad_scaler = runtime.make_scaler(device)
    if initial_grad_scaler_state is not None:
        grad_scaler.load_state_dict(initial_grad_scaler_state)
    reward_replay = build_verifier_replay(
        cli,
        token_runtime,
        taxonomy,
        tolerances,
    )
    epochs = 1 if cli.contract_smoke else int(
        cli.epochs if cli.epochs is not None else training["epochs"]
    )
    history = []
    best_score = None
    cli.output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(epochs):
        if bucket_sampler is not None:
            bucket_sampler.set_epoch(epoch)
        epoch_started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        profile_output = cli.profile_dir if epoch == 0 else None
        with profile_training_epoch(device, profile_output) as profiler:
            train_metrics = run_epoch(
                model,
                train_loader,
                taxonomy,
                device,
                optimizer=optimizer,
                config=cli,
                loss_weights=loss_weights,
                tolerances=tolerances,
                token_runtime=token_runtime,
                runtime=runtime,
                grad_scaler=grad_scaler,
                verifier_replay=reward_replay,
                profiler=profiler,
            )
        with runtime.eager_model(model):
            val_metrics = run_epoch(
                model,
                val_loader,
                taxonomy,
                device,
                optimizer=None,
                config=cli,
                loss_weights=loss_weights,
                tolerances=tolerances,
                token_runtime=token_runtime,
                runtime=runtime,
                grad_scaler=grad_scaler,
                verifier_replay=None,
            )
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        selection_score = rplan_selection_score(
            val_metrics,
            token_enabled=token_runtime.enabled,
            freeze_non_stage=cli.freeze_non_stage,
        )
        row["selection_score"] = selection_score
        elapsed = time.perf_counter() - epoch_started
        row["elapsed_seconds"] = elapsed
        row["train_samples_per_second"] = (
            train_metrics["cases"] / max(elapsed, 1.0e-8)
        )
        if device.type == "cuda":
            row["peak_cuda_memory_bytes"] = int(
                torch.cuda.max_memory_allocated(device)
            )
        history.append(row)
        print(json.dumps(row), flush=True)
        checkpoint = {
            "model": model.state_dict(),
            "taxonomy": taxonomy.to_json(),
            "config": {**config_payload, "model": model_config},
            "epoch": epoch,
            "global_step": token_runtime.step,
            "optimizer": optimizer.state_dict(),
            "grad_scaler": grad_scaler.state_dict(),
            "metrics": val_metrics,
            "token_training": token_runtime.checkpoint_payload(),
            "runtime_acceleration": runtime.manifest(),
            "verifier_reward_replay": (
                reward_replay.statistics()
                if reward_replay is not None
                else None
            ),
        }
        torch.save(checkpoint, cli.output_dir / "last.pt")
        if best_score is None or selection_score > best_score:
            best_score = selection_score
            torch.save(checkpoint, cli.output_dir / "best.pt")
        if cli.contract_smoke:
            break
    if reward_replay is not None:
        reward_replay.close()
        if cli.write_verifier_reward_replay is not None:
            reward_replay.write_replay(cli.write_verifier_reward_replay)
    report = {
        "status": "contract_smoke" if cli.contract_smoke else "complete",
        "test_opened": False,
        "device": str(device),
        "seed": seed,
        "train_cases": len(train_dataset),
        "validation_cases": len(val_dataset),
        "train_offset": cli.train_offset,
        "validation_offset": cli.validation_offset,
        "global_step": token_runtime.step,
        "parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "trainable_parameter_count": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "init_status": init_status,
        "history": history,
        "best_selection_score": best_score,
        "checkpoint_metric": "rplan_gate_balanced_progress_mean",
        "config": asdict(cli),
        "effective_model_config": model_config,
        "effective_loss_weights": loss_weights,
        "token_training": token_runtime.checkpoint_payload(),
        "runtime_acceleration": runtime.manifest(),
        "verifier_reward_replay": (
            reward_replay.statistics()
            if reward_replay is not None
            else None
        ),
        "loss_config_policy": (
            "disabled architecture heads have their auxiliary loss set to zero"
        ),
    }
    cli.output_dir.mkdir(parents=True, exist_ok=True)
    (cli.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
