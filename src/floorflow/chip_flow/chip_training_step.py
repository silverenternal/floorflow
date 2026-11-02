"""Gradient-accumulation loop for chip supervised objectives."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import torch

from floorflow.chip_flow.train_objective import supervised_loss


class BatchStream(Protocol):
    def __iter__(self): ...


def accumulate_supervised_gradients(
    model: torch.nn.Module,
    batch_stream: BatchStream,
    args,
    device: torch.device,
    *,
    step: int,
    data_order_hash,
    response_sidecar=None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    parts_total: dict[str, torch.Tensor] = {}
    loss_total = None
    count = 0
    for batch in batch_stream:
        count += 1
        data_order_hash.update(
            np.asarray(batch["_indices"], dtype=np.int64).tobytes()
        )
        if response_sidecar is not None:
            response_sidecar.attach(batch)
        micro_loss, micro_parts = supervised_loss(
            model,
            batch,
            args,
            device,
            step=step,
        )
        checked_names = ("total", *micro_parts)
        finite = torch.stack(
            (
                torch.isfinite(micro_loss),
                *[
                    torch.isfinite(value).all()
                    for value in micro_parts.values()
                ],
            )
        ).cpu()
        invalid = [
            name
            for name, is_finite in zip(checked_names, finite, strict=True)
            if not bool(is_finite)
        ]
        if invalid:
            raise FloatingPointError(
                f"non-finite training loss terms at step {step}: {invalid}"
            )
        (micro_loss / args.grad_accum_steps).backward()
        detached = micro_loss.detach()
        loss_total = detached if loss_total is None else loss_total + detached
        for name, value in micro_parts.items():
            part = value.detach()
            parts_total[name] = (
                part if name not in parts_total else parts_total[name] + part
            )
    if count != args.grad_accum_steps or loss_total is None:
        raise RuntimeError("gradient accumulation produced an invalid batch count")
    return (
        loss_total / args.grad_accum_steps,
        {
            name: value / args.grad_accum_steps
            for name, value in parts_total.items()
        },
    )


__all__ = ["accumulate_supervised_gradients"]
