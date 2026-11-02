"""Atomic checkpoint persistence for the chip training entry point."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from floorflow.chip_flow.model_config import model_config
from floorflow.chip_flow.train_evaluation import BalancedBatchSampler
from floorflow.chip_flow.training_runtime import (
    cuda_rng_state,
    save_torch_atomic,
)


def save_chip_checkpoint(
    path: Path,
    *,
    step: int,
    model: torch.nn.Module,
    ema: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    batch_rng: np.random.Generator,
    balanced_sampler: BalancedBatchSampler | None,
    data_order_hash: str,
    args,
) -> None:
    """Persist model, optimizer, EMA and all stochastic resume state atomically."""

    save_torch_atomic(
        path,
        {
            "step": step,
            "model": model.state_dict(),
            "ema": ema,
            "optimizer": optimizer.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
            "batch_rng_state": batch_rng.bit_generator.state,
            "cuda_rng_state": cuda_rng_state(args.device),
            "balanced_sampler_state": (
                balanced_sampler.state_dict()
                if balanced_sampler is not None
                else None
            ),
            "data_order_hash": data_order_hash,
            "config": vars(args) | model_config(args),
            "runtime": {
                "fused_adamw": str(args.device).startswith("cuda"),
                "ema": "foreach",
                "projection": "vectorized",
                "gradient_clipping": bool(args.grad_clip_per_group),
            },
        },
    )


__all__ = ["save_chip_checkpoint"]
