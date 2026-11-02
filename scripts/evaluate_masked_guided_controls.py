#!/usr/bin/env python3
"""Evaluate a masked guided CFM across coarse-control ratios."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.evaluation.room_metric_suite import room_metric_suite
from floorflow.flow.condition_view import (
    contour_only_view,
    layout_conditioned_view,
    sparse_control_view,
)
from floorflow.flow.guided_control import (
    guided_source_from_labels,
    sample_guided_floorplans,
)
from floorflow.flow.mask_policies import AttributeMaskPolicy, make_mask_policy
from floorflow.flow.masked_control import apply_control_mask, control_channel_groups
from floorflow.heads.room_program import gate_room_labels_to_program
from floorflow.inference.dual_pass import sample_dual_pass
from floorflow.inference.iterative_control_sampling import sample_iterative_controlled
from floorflow.model.learned_mask_generator import LearnedMaskGenerator
from floorflow.model.sparse_control_attention import (
    MaskedControlFlow,
    sample_sparse_guided_floorplans,
)
from floorflow.training.checkpoint import load_checkpoint_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path(".data/cubicasa5k"))
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("outputs/cache/cubicasa5k_full_v2"),
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=191)
    parser.add_argument("--flow-steps", type=int, default=4)
    parser.add_argument(
        "--ratios",
        nargs="+",
        type=float,
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
    )
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--reveal-fraction", type=float, default=0.25)
    parser.add_argument("--dual-pass", action="store_true")
    parser.add_argument("--remask-ratio", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--mask-policy",
        choices=("spatial", "attribute", "composite", "learned"),
        default="attribute",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "missing_token" not in payload:
        raise ValueError("checkpoint has no learned missing token")
    model, config, taxonomy, _state = load_checkpoint_model(
        args.checkpoint,
        args.device,
        taxonomy=None,
    )
    if "sparse_attention" in payload:
        wrapper = MaskedControlFlow(
            model,
            attention_type=str(
                payload.get("config", {}).get(
                    "sparse_attention_type",
                    "mita",
                )
            ),
        ).to(args.device)
        wrapper.sparse_attention.load_state_dict(
            payload["sparse_attention"],
            strict=False,
        )
        model = wrapper
    if args.dual_pass and not isinstance(model, MaskedControlFlow):
        raise ValueError("dual-pass evaluation requires sparse-control attention")
    mask_generator = (
        LearnedMaskGenerator(model.target_channels).to(args.device)
        if "mask_generator" in payload
        else None
    )
    if mask_generator is not None:
        mask_generator.load_state_dict(payload["mask_generator"])
        mask_generator.eval()
    dataset = CubiCasaFloorplanDataset(
        args.data_root,
        args.split,
        image_size=int(config.get("image_size", 256)),
        cache_root=args.cache_root,
        limit=args.limit,
        augment=False,
    )
    room_ids = torch.tensor(taxonomy.room_ids, device=args.device)
    missing_token = payload["missing_token"].to(args.device)
    summaries = {}
    provenance = {}
    dual_diagnostics = {}
    routing_diagnostics = {}
    if args.mask_policy == "learned" and mask_generator is None:
        raise ValueError("learned mask policy requires a mask_generator checkpoint")
    for ratio in args.ratios:
        rows = []
        for index in range(len(dataset)):
            target, footprint, floor_id, _name = dataset[index]
            target = target[None].to(args.device)
            footprint = footprint[None].to(args.device)
            floor_ids = torch.tensor([floor_id], device=args.device)
            torch.manual_seed(args.seed + index * 1009 + int(ratio * 1000))
            full_source = guided_source_from_labels(target, footprint, taxonomy)
            if ratio <= 0.0:
                condition = layout_conditioned_view(
                    full_source,
                    footprint,
                    floor_ids=floor_ids,
                )
                control_mask = torch.zeros_like(footprint[:, None], dtype=torch.bool)
            elif ratio >= 1.0:
                condition = contour_only_view(
                    footprint,
                    full_source.shape[1],
                    floor_ids=floor_ids,
                )
                control_mask = footprint[:, None].bool()
            elif args.mask_policy == "learned":
                seed_ratio = max(0.01, min(0.75, ratio * 0.5))
                seed_mask = AttributeMaskPolicy().sample(
                    full_source,
                    footprint,
                    taxonomy,
                    ratio=seed_ratio,
                )
                condition = sparse_control_view(
                    full_source,
                    footprint,
                    seed_mask.value,
                    floor_ids=floor_ids,
                )
                generated = mask_generator(
                    condition.control,
                    condition.footprint,
                    target_ratio=ratio,
                    known_mask=seed_mask.value,
                    visibility=condition.visibility,
                    channel_groups=tuple(
                        control_channel_groups(taxonomy)[name].to(args.device)
                        for name in ("room", "wall", "opening", "furniture")
                    ),
                )
                control_mask = generated.hard
            else:
                policy = make_mask_policy(
                    args.mask_policy
                )
                sampled_mask = policy.sample(
                    full_source,
                    footprint,
                    taxonomy,
                    ratio=ratio,
                )
                condition = sparse_control_view(
                    full_source,
                    footprint,
                    sampled_mask.value,
                    floor_ids=floor_ids,
                )
                control_mask = sampled_mask.value
            hint = apply_control_mask(
                condition.control,
                control_mask,
                missing_token,
            )
            provenance.setdefault(str(ratio), condition.manifest())
            if isinstance(model, MaskedControlFlow):
                if args.dual_pass:
                    dual = sample_dual_pass(
                        model,
                        condition.footprint,
                        hint,
                        control_mask,
                        missing_token,
                        steps_per_pass=args.flow_steps,
                        remask_ratio=args.remask_ratio,
                        floor_ids=condition.floor_ids,
                    )
                    sampled = dual.final
                    dual_diagnostics.setdefault(str(ratio), []).append(
                        dual.diagnostics
                    )
                elif args.iterations > 1:
                    sampled, _control, _remaining = sample_iterative_controlled(
                        model,
                        condition.footprint,
                        hint,
                        control_mask,
                        rounds=args.iterations,
                        steps_per_round=args.flow_steps,
                        reveal_fraction=args.reveal_fraction,
                        floor_ids=condition.floor_ids,
                    )
                else:
                    sampled = sample_sparse_guided_floorplans(
                        model,
                        condition.footprint,
                        hint,
                        control_mask,
                        steps=args.flow_steps,
                        floor_ids=condition.floor_ids,
                    )
            else:
                sampled = sample_guided_floorplans(
                    model,
                    condition.footprint,
                    hint,
                    steps=args.flow_steps,
                    floor_ids=condition.floor_ids,
                )
            if isinstance(model, MaskedControlFlow):
                sparse = model.sparse_attention.last_diagnostics
                routing_diagnostics.setdefault(str(ratio), []).append(
                    {
                        key: float(sparse[key].detach().cpu())
                        for key in (
                            "routing_entropy",
                            "effective_expert_fraction",
                            "mask_coverage",
                            "router_budget_mean",
                            "route_balance_loss",
                            "adaptive_mha_weight",
                            "adaptive_control_coverage",
                        )
                        if key in sparse
                    }
                )
            scores = sampled[:, room_ids - 1]
            labels = room_ids[scores.argmax(dim=1)]
            program = scores.amax(dim=(2, 3)) > 0.1
            missing = ~program.any(dim=1)
            if missing.any():
                fallback = scores.amax(dim=(2, 3)).argmax(dim=1)
                program = program.clone()
                program[missing, fallback[missing]] = True
            labels = gate_room_labels_to_program(
                labels,
                program,
                taxonomy.room_ids,
            ) * footprint.long()
            support = torch.isin(labels[0], room_ids)
            suite = room_metric_suite(
                labels[0].cpu().numpy(),
                target[0].cpu().numpy(),
                footprint[0].cpu().numpy(),
                taxonomy,
                predicted_room_support=support.cpu().numpy(),
                predicted_instance_support=support.cpu().numpy(),
                target_room_support=torch.isin(
                    target[0],
                    room_ids,
                ).cpu().numpy(),
            )
            instance = suite["matched_instance"]
            rows.append(
                {
                    "room_macro_iou": suite["visible_semantic"]["room_macro_iou"],
                    "instance_geometric_iou": instance["matched_geometric_iou"],
                    "instance_room_f1": instance["instance_f1_at_threshold"],
                    "instance_type_accuracy": instance["matched_type_accuracy"],
                }
            )
        summaries[str(ratio)] = {
            key: {
                "mean": float(np.mean([row[key] for row in rows])),
                "median": float(np.median([row[key] for row in rows])),
                "standard_error": float(
                    np.std([row[key] for row in rows], ddof=1)
                    / np.sqrt(len(rows))
                ),
            }
            for key in rows[0]
        }
        print(f"ratio={ratio:.2f} {json.dumps(summaries[str(ratio)])}", flush=True)
    report = {
        "checkpoint": str(args.checkpoint),
        "cases": len(dataset),
        "flow_steps": args.flow_steps,
        "iterations": args.iterations,
        "reveal_fraction": args.reveal_fraction,
        "mask_policy": args.mask_policy,
        "control_ratio_sweep": summaries,
        "condition_provenance": provenance,
        "dual_pass_diagnostics": {
            ratio: {
                key: float(np.mean([row[key] for row in rows]))
                for key in rows[0]
            }
            for ratio, rows in dual_diagnostics.items()
        },
        "routing_diagnostics": {
            ratio: {
                key: float(np.mean([row[key] for row in rows]))
                for key in rows[0]
            }
            for ratio, rows in routing_diagnostics.items()
        },
        "dual_pass": args.dual_pass,
        "remask_ratio": args.remask_ratio if args.dual_pass else None,
        "best_of_n_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
