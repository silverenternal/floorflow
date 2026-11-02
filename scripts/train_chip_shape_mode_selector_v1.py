#!/usr/bin/env python3
"""Train only the chip shape-mode selector with legalized teacher labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party/Trinity/src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_chip_official import case_payload, score_case

from floorflow.chip_flow.cache import PackedCaseStore
from floorflow.chip_flow.checkpoint_loading import load_chip_checkpoint
from floorflow.chip_flow.flow_adapter import sample_for_batch
from floorflow.chip_flow.legalizer_policy import LegalizerPolicy
from floorflow.chip_flow.model import decode_boxes_batch
from floorflow.chip_flow.projections import project_conditioned_latent
from floorflow.chip_flow.shape_mode_features import (
    shape_mode_geometry_features,
)
from floorflow.chip_flow.training_runtime import (
    save_torch_atomic,
    write_json_atomic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT
        / "outputs/chip_hierarchy/s7_shape_mode_v1_quick/"
        "shape_mode_v1/checkpoints/final.pt",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=ROOT / "outputs/cache/chip_floorset_50k_train_12k_dev_v2/train",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/chip_hierarchy/shape_mode_selector_v1/best.pt",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "outputs/reports/chip_shape_mode_selector_train_v1.json",
    )
    parser.add_argument("--cases-per-count", type=int, default=4)
    parser.add_argument("--counts", type=int, nargs="+", default=[21, 25, 30, 35, 40, 45, 50, 55])
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--sampling-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _teacher_rows(
    model,
    baseline: torch.Tensor,
    batch: dict,
    context,
    policy: LegalizerPolicy,
) -> tuple[list[int], list[dict]]:
    modes = model.shape_mode_head.forward_with_latents(
        baseline,
        context.typed.block_tokens,
        baseline,
        constraints=context.constraints,
        membership=context.coarse_membership,
        structural_tokens=context.typed.structural_tokens,
        structural_mask=context.typed.structural_mask,
        global_token=context.typed.global_token[:, 0],
    ).mode_latents
    candidates = torch.cat([baseline[:, None], modes], dim=1)
    winners = []
    rows = []
    for case_index in range(baseline.shape[0]):
        costs = []
        for candidate_index in range(candidates.shape[1]):
            latent = project_conditioned_latent(
                candidates[case_index, candidate_index][None],
                batch["target_latent"][case_index : case_index + 1],
                batch["constraints"][case_index : case_index + 1],
            )[0]
            boxes = decode_boxes_batch(
                latent[None],
                batch["areas"][case_index : case_index + 1],
            )
            payload = case_payload(
                batch,
                case_index,
                boxes[0],
                0.0,
                legalize=True,
                legalizer_time_limit_ms=policy.time_limit_ms,
                legalizer_options=policy.kwargs(),
            )
            row = score_case(payload)
            cost = float(row["legalized_hard_cost"])
            if not row["legalized_feasible"]:
                cost = float("inf")
            costs.append(cost)
            rows.append(
                {
                    "case": int(batch["_indices"][case_index]),
                    "mode": candidate_index - 1,
                    "cost": cost,
                    "v_rel": float(row["legalized_v_rel"]),
                }
            )
        winners.append(int(np.argmin(np.asarray(costs))))
    return winners, rows


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    store = PackedCaseStore(args.cache)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = load_chip_checkpoint(checkpoint, str(device), use_ema=False)
    if not hasattr(model, "shape_mode_head"):
        raise RuntimeError("checkpoint has no shape-mode head")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable = [
        *model.shape_mode_head.mode_score.parameters(),
        *model.shape_mode_head.mode_feature_encoder.parameters(),
        *model.shape_mode_head.baseline_feature_score.parameters(),
        model.shape_mode_head.baseline_bias,
        model.shape_mode_head.mode_bias,
    ]
    for parameter in trainable:
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
    policy = LegalizerPolicy(
        relation="restored",
        reshape=True,
        trust=1.3,
        outline=True,
        time_limit_ms=2000,
    )
    policy.validate()
    selected = []
    for count in args.counts:
        indices = store.indices(count)
        if len(indices):
            selected.extend(indices[: args.cases_per_count].tolist())
    history = []
    for epoch in range(args.epochs):
        losses = []
        accuracy = []
        for count in args.counts:
            rows = [index for index in selected if int(store.arrays["n"][index]) == count]
            if not rows:
                continue
            batch = store.batch(np.asarray(rows), device=device)
            batch["_indices"] = torch.as_tensor(rows, device=device)
            with torch.no_grad():
                baseline, _ = sample_for_batch(
                    model,
                    batch,
                    sampling_steps=args.sampling_steps,
                    generator=torch.Generator(device=device).manual_seed(args.seed + epoch),
                )
                context = model.prepare_context(batch)
            winners, teacher_rows = _teacher_rows(
                model,
                baseline,
                batch,
                context,
                policy,
            )
            candidate_features = shape_mode_geometry_features(
                model.shape_mode_head.forward_with_latents(
                    baseline,
                    context.typed.block_tokens,
                    baseline,
                    constraints=context.constraints,
                    membership=context.coarse_membership,
                    structural_tokens=context.typed.structural_tokens,
                    structural_mask=context.typed.structural_mask,
                    global_token=context.typed.global_token[:, 0],
                ).mode_latents,
                baseline,
                batch,
                include_baseline=True,
            )
            mode_features = candidate_features[:, 1:]
            logits = model.shape_mode_head.forward_with_latents(
                baseline,
                context.typed.block_tokens,
                baseline,
                constraints=context.constraints,
                membership=context.coarse_membership,
                structural_tokens=context.typed.structural_tokens,
                structural_mask=context.typed.structural_mask,
                global_token=context.typed.global_token[:, 0],
                mode_features=mode_features,
            ).mode_logits
            baseline_logit = model.shape_mode_head.baseline_logit(
                candidate_features[:, 0]
            )
            logits = torch.cat([baseline_logit[:, None], logits], dim=-1)
            target = torch.tensor(winners, device=device)
            loss = F.cross_entropy(logits.float(), target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            accuracy.append(
                float((logits.argmax(dim=-1) == target).float().mean())
            )
            history.append(
                {
                    "epoch": epoch,
                    "blocks": count,
                    "loss": losses[-1],
                    "accuracy": accuracy[-1],
                    "teacher_rows": teacher_rows,
                }
            )
            print(json.dumps(history[-1]), flush=True)
    payload = {
        **checkpoint,
        "model": model.state_dict(),
        "ema": model.state_dict(),
        "config": {
            **checkpoint.get("config", {}),
            "shape_mode_apply": True,
            "shape_mode_selector_training": "legalized-cost-wta-v1",
        },
    }
    save_torch_atomic(args.output, payload)
    report = {
        "status": "complete",
        "scope": "selector-only CPU training; not a model SOTA result",
        "checkpoint": str(args.checkpoint),
        "output": str(args.output),
        "cases": sorted(set(selected)),
        "epochs": args.epochs,
        "sampling_steps": args.sampling_steps,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "history": history,
        "final_accuracy": float(np.mean(accuracy)),
        "final_loss": float(np.mean(losses)),
    }
    write_json_atomic(args.report, report)
    print(json.dumps({"output": str(args.output), "accuracy": report["final_accuracy"]}))


if __name__ == "__main__":
    main()
