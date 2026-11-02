"""CPU-only audit of grouped optimizer, clipping and EMA contracts."""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path

import torch

from floorflow.chip_flow.cache import PackedCaseStore
from floorflow.chip_flow.hierarchical_flow import (
    HierarchicalTokenFlowModel,
    build_hierarchy_targets,
)
from floorflow.chip_flow.training_runtime import (
    clip_optimizer_gradients_fail_closed,
    ema_decay_at_step,
    initial_ema_state,
    make_optimizer,
    set_optimizer_group_lrs,
    update_ema_state_,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("outputs/cache/chip_floorset_50k_train_12k_dev_v2/dev"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reports/chip_optimizer_contract_audit_v1.json"),
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    store = PackedCaseStore(args.cache)
    block_count = next(
        int(count)
        for count, indices in sorted(store.by_block_count.items())
        if len(indices) >= 2
    )
    batch = store.batch(store.indices(block_count)[:2], device="cpu")
    model = HierarchicalTokenFlowModel(
        hidden=16,
        heads=2,
        depth=1,
        structured_heads=True,
    )
    optimizer = make_optimizer(
        model,
        lr=2.0e-4,
        weight_decay=1.0e-3,
        fused=False,
        backbone_lr_scale=0.5,
        structured_head_lr_scale=0.5,
        legalizer_head_lr_scale=0.25,
        structured_head_warmup_steps=200,
        legalizer_head_warmup_steps=100,
    )
    context = model.prepare_context(batch)
    target = build_hierarchy_targets(batch, context)
    prediction = model(target, torch.full((2,), 0.5), context)
    loss = prediction.block_state.square().mean()
    loss = loss + prediction.port_state.square().mean()
    loss.backward()
    clip = clip_optimizer_gradients_fail_closed(
        model,
        optimizer,
        max_norm=1.0,
        per_group=True,
    )
    lrs_at_zero = set_optimizer_group_lrs(optimizer, 2.0e-4, step=0)
    ema = initial_ema_state(model)
    ema_args = Namespace(
        ema_decay=0.9995,
        ema_mode="warmup",
        ema_warmup_steps=1000,
    )
    decay_zero = ema_decay_at_step(ema_args, 0)
    decay_full = ema_decay_at_step(ema_args, 1000)
    update_ema_state_(ema, model.state_dict(), decay_zero)
    groups = []
    for group in optimizer.param_groups:
        groups.append(
            {
                "name": group["group_name"],
                "parameters": sum(
                    parameter.numel() for parameter in group["params"]
                ),
                "lr_scale": float(group["lr_scale"]),
                "weight_decay": float(group["weight_decay"]),
                "warmup_steps": int(group["warmup_steps"]),
            }
        )
    report = {
        "version": "chip-optimizer-contract-audit-v1",
        "status": "pass",
        "device": "cpu",
        "block_count": block_count,
        "parameter_groups": groups,
        "step_zero_lrs": lrs_at_zero,
        "gradient_clipping": clip,
        "ema": {
            "base_decay": 0.9995,
            "mode": "warmup",
            "warmup_steps": 1000,
            "decay_at_zero": decay_zero,
            "decay_after_warmup": decay_full,
        },
        "boundary": "CPU contract audit only; no GPU or quality claim.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
