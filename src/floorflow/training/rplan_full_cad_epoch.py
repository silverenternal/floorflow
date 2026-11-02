"""Epoch loop for the unified RPLAN boundary-to-CAD generator."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from floorflow.evaluation.fevp_metric_contracts import FEVPTolerances
from floorflow.evaluation.rplan_mask_partition import (
    mask_partition_query_metrics,
)
from floorflow.heads.polygon_loop import room_query_metrics
from floorflow.model.unified_floorplan_generator import UnifiedFloorplanGenerator
from floorflow.objectives.gradient_conflict import conflict_aware_backward
from floorflow.training.floorplan_token_training_runtime import (
    FloorplanTokenTrainingRuntime,
)
from floorflow.training.gpu_checks import (
    assert_finite_async,
    assert_parameters_finite,
)
from floorflow.training.metric_accumulator import MetricAccumulator
from floorflow.training.parameter_scope import scoped_parameters
from floorflow.training.rplan_generator_cli import TrainConfig
from floorflow.training.rplan_generator_contract import contract_sample
from floorflow.training.runtime_acceleration import RuntimeConfig
from floorflow.training.token_consumer_validation import (
    aggregate_token_consumer_reports,
    token_consumer_validation,
)
from floorflow.training.token_validation import token_validation_readout
from floorflow.training.unified_plan_generator_training import (
    unified_generator_loss,
)
from floorflow.training.verifier_reward_feedback import attach_verifier_rewards
from floorflow.training.verifier_reward_replay import VerifierRewardReplay


def _batch_to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: (
            value.to(device, non_blocking=device.type == "cuda")
            if torch.is_tensor(value)
            else value
        )
        for key, value in batch.items()
    }


def run_epoch(
    model: UnifiedFloorplanGenerator,
    loader: DataLoader,
    taxonomy,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None,
    config: TrainConfig,
    loss_weights: dict[str, float],
    tolerances: FEVPTolerances,
    token_runtime: FloorplanTokenTrainingRuntime,
    runtime: RuntimeConfig | None = None,
    grad_scaler: torch.amp.GradScaler | None = None,
    verifier_replay: VerifierRewardReplay | None = None,
    profiler=None,
) -> dict:
    training = optimizer is not None
    model.train(training)
    if training and config.freeze_non_stage:
        model.eval()
        model.vector_decoder.stage_conditioners.train()
    totals = MetricAccumulator()
    seen = 0
    contract = None
    consumer_reports = []
    gradient_report = None
    conflict_report = None
    generation_totals: dict[str, float] = {}
    generation_seen = 0
    for batch_index, raw_batch in enumerate(loader):
        if (
            token_runtime.enabled
            and token_runtime.config.zero_oov
            and "token_oov_count" in raw_batch
            and int(raw_batch["token_oov_count"].sum()) > 0
        ):
            raise ValueError("token target vocabulary has OOV entries")
        batch = _batch_to_device(raw_batch, device)
        execution_runtime = runtime or RuntimeConfig()
        with torch.set_grad_enabled(training), execution_runtime.autocast(device):
            output = model(
                batch["boundary_mask"],
                batch["boundary_valid"],
                batch["boundary_points"],
                batch["entrance_points"],
                batch["entrance_valid"],
                **token_runtime.forward_kwargs(batch, training=training),
            )
            if training and token_runtime.config.verifier_policy_weight > 0.0:
                if verifier_replay is not None:
                    verifier_replay.maybe_refresh(
                        output,
                        batch,
                        token_runtime.step,
                    )
                    reward_batch = verifier_replay.rewards(
                        batch,
                        token_runtime.step,
                        device=device,
                    )
                    batch["verifier_reward"] = reward_batch.rewards
                    batch["verifier_reward_mask"] = reward_batch.valid
                else:
                    batch["verifier_reward"] = attach_verifier_rewards(
                        output,
                        batch,
                        taxonomy,
                        tolerances,
                    )
            loss_kwargs = token_runtime.loss_kwargs()
            if token_runtime.config.hierarchy_weights:
                loss_kwargs["hierarchy_weights"] = dict(
                    token_runtime.config.hierarchy_weights
                )
            if token_runtime.config.verifier_policy_weight > 0.0:
                loss_kwargs["verifier_policy_weight"] = (
                    token_runtime.config.verifier_policy_weight
                )
                loss_kwargs["verifier_reward_baseline"] = (
                    token_runtime.config.verifier_reward_baseline
                )
            loss_terms: dict[str, torch.Tensor] = {}
            if token_runtime.config.conflict_projection:
                loss_kwargs["loss_terms_out"] = loss_terms
            if training and token_runtime.loss_balancer is not None:
                loss_kwargs.update(
                    balance_parameters=scoped_parameters(
                        model,
                        token_runtime.config.balance_parameter_scope,
                    ),
                    balance_step=token_runtime.step,
                )
            if (
                training
                and batch_index == 0
                and bool(token_runtime.config.balance.get("diagnostics", False))
            ):
                gradient_report = {}
                loss_kwargs.update(
                    diagnostic_parameters=list(model.parameters()),
                    diagnostic_out=gradient_report,
                )
            loss, components = unified_generator_loss(
                output,
                batch,
                taxonomy,
                polygon_weight=float(loss_weights["polygon"]),
                joint_mask_match_weight=(
                    float(config.joint_mask_match_weight)
                    if config.joint_mask_match_weight is not None
                    else 1.0
                ),
                joint_room_assignment=config.enable_joint_room_assignment,
                identity_weight=float(loss_weights["identity"]),
                count_weight=float(loss_weights["count"]),
                wall_weight=float(loss_weights["wall"]),
                opening_weight=float(loss_weights["opening"]),
                adjacency_weight=float(loss_weights["adjacency"]),
                query_separation_weight=float(
                    config.query_separation_weight
                    if config.query_separation_weight is not None
                    else loss_weights.get("query_separation", 0.0)
                ),
                class_balance_power=float(
                    config.class_balance_power
                    if config.class_balance_power is not None
                    else loss_weights.get("class_balance_power", 0.0)
                ),
                class_aware_separation=bool(
                    config.class_aware_separation
                    or loss_weights.get("class_aware_separation", False)
                ),
                supervised_separation=bool(
                    config.supervised_separation
                    or loss_weights.get("supervised_separation", False)
                ),
                query_quality_weight=float(
                    config.query_quality_weight
                    if config.query_quality_weight is not None
                    else loss_weights.get("query_quality", 0.0)
                ),
                point_validity_weight=float(
                    config.point_validity_weight
                    if config.point_validity_weight is not None
                    else loss_weights.get("point_validity", 0.5)
                ),
                room_program_weight=float(
                    config.room_program_weight
                    if config.room_program_weight is not None
                    else loss_weights.get("room_program", 0.0)
                ),
                partition_consistency_weight=float(
                    config.partition_consistency_weight
                    if config.partition_consistency_weight is not None
                    else loss_weights.get("partition_consistency", 0.0)
                ),
                partition_conflict_only=bool(
                    loss_weights.get("partition_conflict_only", False)
                ),
                partition_label_smoothing=float(
                    loss_weights.get("partition_label_smoothing", 0.0)
                ),
                **loss_kwargs,
            )
        assert_finite_async(loss, "non-finite RPLAN generator loss")
        if optimizer is not None:
            scaler = grad_scaler or execution_runtime.make_scaler(device)
            optimizer.zero_grad(set_to_none=True)
            use_conflict = (
                token_runtime.config.conflict_projection
                and token_runtime.step % token_runtime.config.conflict_interval == 0
            )
            if use_conflict:
                trainable_parameters = scoped_parameters(
                    model,
                    token_runtime.config.conflict_parameter_scope,
                )
                stats = conflict_aware_backward(
                    loss_terms,
                    trainable_parameters,
                    token_runtime.config.conflict_pairs,
                    loss_scale=(
                        float(scaler.get_scale())
                        if scaler.is_enabled()
                        else 1.0
                    ),
                    total_loss=scaler.scale(loss),
                )
                if batch_index == 0:
                    conflict_report = stats
            else:
                scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            assert_parameters_finite(
                model.parameters(),
                "non-finite RPLAN generator gradient",
            )
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            token_runtime.step += 1
        batch_size = batch["boundary_mask"].shape[0]
        totals.add("loss", loss, weight=batch_size)
        for key, value in components.items():
            totals.add(key, value, weight=batch_size)
        predicted_count = output["room_count_logits"].argmax(dim=-1)
        totals.add(
            "room_count_mae",
            (predicted_count - batch["room_count"]).abs().sum(),
        )
        totals.add(
            "room_count_correct",
            (predicted_count == batch["room_count"]).sum(),
        )
        if "token_length" in batch:
            totals.add("token_length", batch["token_length"].sum())
            totals.add("token_oov_count", batch["token_oov_count"].sum())
        for key, value in output.items():
            if key.startswith(
                ("token_refiner_", "token_graph_", "codetree_")
            ) and torch.is_tensor(value) and torch.is_floating_point(value):
                totals.add(key, value, weight=batch_size)
        if not training:
            token_metrics, generated = token_validation_readout(
                model,
                output,
                batch,
                token_runtime,
                collect_generation=False,
            )
            for key, value in token_metrics.items():
                totals.add(key, value, weight=batch_size)
            if generated is not None:
                generation_seen += 1
                for key, value in generated.items():
                    generation_totals[key] = (
                        generation_totals.get(key, 0.0) + float(value)
                    )
            query_metrics = room_query_metrics(
                {
                    "class_logits": output["room_class"],
                    "polygons": output["room_polygons"],
                    "valid_logits": output["room_exists"],
                },
                batch["target_labels"],
                taxonomy,
                validity_threshold=0.25,
                min_area=4,
                query_budget=predicted_count,
                query_quality_logits=output.get("query_quality_logits"),
            )
            partition_metrics = mask_partition_query_metrics(
                output,
                batch["target_labels"],
                batch["boundary_mask"],
                taxonomy,
            )
            for key, value in partition_metrics.items():
                totals.add(
                    f"mask_partition_{key}",
                    value,
                    weight=batch_size,
                )
            if config.readout == "mask_partition":
                query_metrics = partition_metrics
            for key, value in query_metrics.items():
                totals.add(key, value, weight=batch_size)
        seen += batch_size
        if not training:
            consumer_samples = sum(
                len(report[1]["vector"]) for report in consumer_reports
            )
            if (
                token_runtime.enabled
                and consumer_samples < token_runtime.config.generation_eval_samples
            ):
                consumer_metrics, consumer_details = token_consumer_validation(
                    model,
                    batch,
                    token_runtime,
                    taxonomy,
                    tolerances,
                    max_samples=(
                        token_runtime.config.generation_eval_samples
                        - consumer_samples
                    ),
                )
                consumer_reports.append((consumer_metrics, consumer_details))
                if (
                    token_runtime.config.hard_validity_regression_gate
                    and (
                        consumer_metrics.get(
                            "token_consumer_teacher_validity_regression",
                            0.0,
                        )
                        > 0.0
                        or (
                            consumer_metrics.get(
                                "token_consumer_generation_completed",
                                0.0,
                            )
                            > 0.0
                            and consumer_metrics.get(
                                "token_consumer_generated_validity_regression",
                                0.0,
                            )
                            > 0.0
                        )
                        or consumer_metrics.get(
                            "token_consumer_teacher_watertight_regression",
                            0.0,
                        )
                        > 0.0
                        or (
                            consumer_metrics.get(
                                "token_consumer_generation_completed",
                                0.0,
                            )
                            > 0.0
                            and consumer_metrics.get(
                                "token_consumer_generated_watertight_regression",
                                0.0,
                            )
                            > 0.0
                        )
                    )
                ):
                    raise RuntimeError(
                        "token refinement regressed canonical validity"
                    )
            elif config.contract_smoke and batch_index == 0:
                contract = contract_sample(
                    model,
                    batch,
                    taxonomy,
                    tolerances,
                    token_runtime,
                )
        if profiler is not None:
            profiler.step()
    metrics = totals.finalize(seen)
    metrics["cases"] = seen
    if gradient_report is not None:
        metrics["gradient_diagnostics"] = gradient_report
    if conflict_report is not None:
        metrics["gradient_conflict"] = conflict_report
    if generation_seen:
        metrics["token_generation"] = {
            key: value / generation_seen
            for key, value in generation_totals.items()
        }
    if consumer_reports:
        consumer_metrics = aggregate_token_consumer_reports(consumer_reports)
        metrics["token_consumer"] = consumer_metrics
        metrics["token_generation"] = {
            "token_generation_completed": consumer_metrics.get(
                "token_consumer_generation_completed",
                0.0,
            ),
            "token_generated_count": consumer_metrics.get(
                "token_consumer_generation_count",
                0.0,
            ),
            "token_generation_within_efficiency_budget": (
                consumer_metrics.get(
                    "token_consumer_generation_within_efficiency_budget",
                    0.0,
                )
            ),
        }
        combined = {
            branch: [
                item
                for _metrics, details in consumer_reports
                for item in details[branch]
            ]
            for branch in ("vector", "teacher", "generated")
        }
        contract = combined["teacher"][0] if combined["teacher"] else None
    if contract is not None:
        metrics["contract"] = contract
    return metrics


__all__ = ["run_epoch"]
