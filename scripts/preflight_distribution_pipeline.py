#!/usr/bin/env python3
"""Fail-fast overfit and train/eval path audit before full training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader

from floorflow.data.class_balance import load_positive_weights
from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.distribution.cfm_model import DistributionCFMModel
from floorflow.distribution.checkpoint import save_distribution_model
from floorflow.distribution.element_loss import element_set_loss
from floorflow.distribution.element_targets import element_targets
from floorflow.distribution.hierarchy import ChannelHierarchy
from floorflow.distribution.joint_loss import joint_distribution_cfm_loss
from floorflow.objectives.condition_objective import room_macro_iou_from_labels
from floorflow.training.checkpoint import load_checkpoint_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path(".data/cubicasa5k"))
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("outputs/cache/cubicasa5k_full_v2"),
    )
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--element-warmup-steps", type=int, default=700)
    parser.add_argument("--element-warmup-lr", type=float, default=1.0e-3)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--min-loss-drop", type=float, default=0.10)
    parser.add_argument("--min-element-iou", type=float, default=0.20)
    parser.add_argument(
        "--element-mask-version",
        choices=("v1", "v2"),
        default="v2",
    )
    parser.add_argument("--save-final", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def room_score(model, target, footprint, floor_ids, taxonomy):
    output = model.contour_initial_state(
        footprint,
        floor_ids=floor_ids,
    )
    prediction = output["pixel"]["logits"].argmax(dim=1) * footprint[:, 0].long()
    return float(
        room_macro_iou_from_labels(prediction, target, taxonomy)
    )


@torch.inference_mode()
def element_score(
    model,
    target,
    footprint,
    floor_ids,
    taxonomy,
) -> dict[str, float]:
    output = model.contour_initial_state(footprint, floor_ids=floor_ids)
    targets = element_targets(
        target,
        taxonomy,
        model.hub.hierarchy,
        max_queries=model.hub.element.query_counts,
        min_area=8,
    )
    report = {}
    for name, prediction in output["elements"].items():
        family_target = targets[name]
        ious = []
        probability = prediction["mask_logits"].sigmoid()
        type_probability = prediction["type_logits"].softmax(dim=-1)
        for batch_index in range(probability.shape[0]):
            count = int(family_target.valid[batch_index].sum())
            if count == 0:
                continue
            expected = family_target.masks[batch_index, :count]
            if expected.shape[-2:] != probability.shape[-2:]:
                expected = F.interpolate(
                    expected[:, None],
                    size=probability.shape[-2:],
                    mode="nearest",
                )[:, 0]
            predicted = probability[batch_index].flatten(1)
            expected_flat = expected.flatten(1)
            intersection = predicted @ expected_flat.T
            union = (
                predicted.sum(dim=1)[:, None]
                + expected_flat.sum(dim=1)[None]
                - intersection
            )
            mask_cost = 1.0 - intersection / union.clamp_min(1.0e-6)
            class_cost = -type_probability[
                batch_index,
                :,
                family_target.types[batch_index, :count],
            ]
            rows, columns = linear_sum_assignment(
                (mask_cost + class_cost).detach().cpu().numpy()
            )
            for row, column in zip(rows, columns):
                hard_prediction = probability[batch_index, row] >= 0.5
                hard_expected = expected[column] >= 0.5
                hard_union = (hard_prediction | hard_expected).sum()
                ious.append(
                    float(
                        (hard_prediction & hard_expected).sum()
                        / hard_union.clamp_min(1)
                    )
                )
        if ious:
            report[name] = float(np.mean(ious))
    return report


@torch.inference_mode()
def element_diagnostics(
    model,
    footprint,
    floor_ids,
) -> dict[str, dict[str, float]]:
    output = model.contour_initial_state(footprint, floor_ids=floor_ids)
    return {
        name: {
            key: float(prediction[key])
            for key in (
                "mask_pairwise_iou",
                "mask_positive_fraction",
                "mask_mean_probability",
                "mask_logit_std",
            )
        }
        for name, prediction in output["elements"].items()
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(19)
    device = torch.device(args.device)
    base, config, taxonomy, _ = load_checkpoint_model(
        args.checkpoint,
        device,
        taxonomy=None,
    )
    hierarchy = ChannelHierarchy.from_taxonomy(taxonomy)
    counts = {
        "wall": 8,
        "opening": 8,
        "room": 8,
        "furniture": 8,
        "support": 4,
    }
    model = DistributionCFMModel(
        base,
        hierarchy,
        hidden_channels=96,
        pixel_grid=8,
        element_grid=8,
        element_resolution=64,
        queries_per_group=counts,
        element_mask_version=args.element_mask_version,
    ).to(device)
    base.eval().requires_grad_(False)
    dataset = CubiCasaFloorplanDataset(
        args.data_root,
        "train",
        image_size=int(config.get("image_size", 256)),
        cache_root=args.cache_root,
        limit=2,
        augment=False,
    )
    target, footprint, floor_ids, _ = next(
        iter(DataLoader(dataset, batch_size=2, shuffle=False))
    )
    target = target.to(device)
    footprint = footprint[:, None].to(device)
    floor_ids = floor_ids.to(device)
    positive = torch.from_numpy(
        load_positive_weights(
            args.cache_root,
            taxonomy.num_target_channels,
            20.0,
        )
    ).to(device)
    weights = torch.cat([torch.ones(1, device=device), positive])
    element_targets_by_family = element_targets(
        target,
        taxonomy,
        hierarchy,
        max_queries=counts,
        min_area=8,
    )
    element_optimizer = torch.optim.AdamW(
        model.hub.element.parameters(),
        lr=args.element_warmup_lr,
        weight_decay=1.0e-4,
    )
    element_warmup_loss = None
    model.train()
    model.hub.pixel.eval()
    for _step in range(max(0, args.element_warmup_steps)):
        initial = model.contour_initial_state(
            footprint,
            floor_ids=floor_ids,
        )
        element_warmup_loss, _components = element_set_loss(
            initial["elements"],
            element_targets_by_family,
        )
        element_optimizer.zero_grad(set_to_none=True)
        element_warmup_loss.backward()
        element_optimizer.step()
    model.train()
    parameters = list(model.hub.parameters()) + list(model.cfm_bridge.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=1.0e-4)
    with torch.no_grad():
        first = model.contour_initial_state(footprint, floor_ids=floor_ids)
        second = model.contour_initial_state(footprint, floor_ids=floor_ids)
        deterministic = torch.equal(
            first["pixel"]["logits"],
            second["pixel"]["logits"],
        )
        initial_score = room_score(
            model,
            target,
            footprint,
            floor_ids,
            taxonomy,
        )
        initial_elements = element_score(
            model,
            target,
            footprint,
            floor_ids,
            taxonomy,
        )
    initial_loss = None
    final_loss = None
    final_components = {}
    for step in range(max(1, args.steps)):
        model.train()
        initial = model.contour_initial_state(
            footprint,
            floor_ids=floor_ids,
        )
        loss, components = joint_distribution_cfm_loss(
            model,
            target,
            initial["cfm"]["state"],
            footprint,
            taxonomy,
            hierarchy,
            floor_ids=floor_ids,
            queries_per_group=counts,
            class_weights=weights,
            contour_output=initial,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        value = float(loss.detach())
        initial_loss = value if initial_loss is None else initial_loss
        final_loss = value
        final_components = {
            key: float(value)
            for key, value in components.items()
            if "mask" in key or "element" in key
        }
    with torch.no_grad():
        final_score = room_score(
            model,
            target,
            footprint,
            floor_ids,
            taxonomy,
        )
        final_elements = element_score(
            model,
            target,
            footprint,
            floor_ids,
            taxonomy,
        )
        final_diagnostics = element_diagnostics(
            model,
            footprint,
            floor_ids,
        )
    loss_drop = (
        (float(initial_loss) - float(final_loss))
        / max(abs(float(initial_loss)), 1.0e-6)
    )
    report = {
        "checkpoint": str(args.checkpoint),
        "steps": args.steps,
        "element_warmup_steps": args.element_warmup_steps,
        "element_warmup_loss": (
            float(element_warmup_loss.detach())
            if element_warmup_loss is not None
            else None
        ),
        "deterministic_contour_path": deterministic,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_drop_fraction": loss_drop,
        "initial_room_macro_iou": initial_score,
        "final_room_macro_iou": final_score,
        "room_iou_gain": final_score - initial_score,
        "element_mask_version": args.element_mask_version,
        "initial_element_iou": initial_elements,
        "final_element_iou": final_elements,
        "element_iou_gain": {
            name: final_elements[name] - initial_elements.get(name, 0.0)
            for name in final_elements
        },
        "final_element_diagnostics": final_diagnostics,
        "final_element_components": final_components,
        "minimum_final_element_iou": (
            min(final_elements.values()) if final_elements else 0.0
        ),
        "element_gate_passed": bool(
            final_elements
            and min(final_elements.values()) >= args.min_element_iou
        ),
        "passed": bool(
            deterministic
            and loss_drop >= args.min_loss_drop
            and final_score >= initial_score
            and final_elements
            and min(final_elements.values()) >= args.min_element_iou
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.save_final is not None:
        save_distribution_model(
            model,
            args.save_final,
            taxonomy=taxonomy,
            extra={"preflight": report},
        )
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
