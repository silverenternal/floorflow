"""Runtime and asynchronous verifier setup for RPLAN training."""

from __future__ import annotations

import torch

from floorflow.evaluation.fevp_metric_contracts import FEVPTolerances
from floorflow.training.floorplan_token_training_runtime import (
    FloorplanTokenTrainingRuntime,
)
from floorflow.training.rplan_generator_cli import TrainConfig
from floorflow.training.runtime_acceleration import (
    RuntimeConfig,
    configure_torch_runtime,
)
from floorflow.training.verifier_reward_replay import VerifierRewardReplay


def build_acceleration_runtime(
    cli: TrainConfig,
    config_payload: dict,
    device: torch.device,
) -> RuntimeConfig:
    payload = dict(config_payload.get("runtime") or {})
    runtime = RuntimeConfig(
        precision=cli.precision or str(payload.get("precision", "fp32")),
        compile_model=bool(
            cli.compile_model or payload.get("compile_model", False)
        ),
        compile_mode=str(
            cli.compile_mode
            if cli.compile_mode != "default"
            else payload.get("compile_mode", "default")
        ),
        channels_last=bool(
            cli.channels_last or payload.get("channels_last", False)
        ),
        persistent_workers=bool(payload.get("persistent_workers", True)),
        prefetch_factor=int(
            cli.prefetch_factor
            if cli.prefetch_factor is not None
            else payload.get("prefetch_factor", 4)
        ),
        tf32=bool(cli.tf32 and payload.get("tf32", True)),
        triton_kernels=bool(
            payload.get("triton_kernels", False)
            and not cli.disable_triton_kernels
        ),
    )
    configure_torch_runtime(
        device,
        tf32=runtime.tf32,
        triton_kernels=runtime.triton_kernels,
    )
    return runtime


def build_verifier_replay(
    cli: TrainConfig,
    token_runtime: FloorplanTokenTrainingRuntime,
    taxonomy,
    tolerances: FEVPTolerances,
) -> VerifierRewardReplay | None:
    if (
        not token_runtime.enabled
        or token_runtime.config.verifier_policy_weight <= 0.0
    ):
        return None
    payload = dict(token_runtime.config.verifier_reward_replay or {})
    namespace = ":".join(
        (
            "floorflow-verifier-reward-v1",
            token_runtime.vocabulary.sha256(),
            str(token_runtime.config.max_length),
            str(token_runtime.config.numeric_bin_size),
            str(token_runtime.config.geometry_token_mode),
        )
    )
    return VerifierRewardReplay(
        taxonomy,
        tolerances,
        namespace=namespace,
        refresh_interval=int(
            cli.verifier_reward_refresh_interval
            or payload.get("refresh_interval", 20)
        ),
        max_age_steps=int(
            cli.verifier_reward_max_age_steps
            or payload.get("max_age_steps", 200)
        ),
        max_workers=int(payload.get("workers", 1)),
        replay_path=cli.verifier_reward_replay or payload.get("path"),
    )


__all__ = ["build_acceleration_runtime", "build_verifier_replay"]
