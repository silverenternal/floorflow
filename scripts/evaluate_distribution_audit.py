#!/usr/bin/env python3
"""Audit per-family pixel, element, and hierarchy diagnostics."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, Subset

from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.distribution.checkpoint import restore_distribution_model
from floorflow.distribution.element_targets import element_targets


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
    parser.add_argument("--offset", type=int, default=191)
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--disable-parent-reliability", action="store_true")
    parser.add_argument("--hierarchy-support-strength", type=float, default=None)
    return parser.parse_args()


def _dice_to_iou(mask_probability: torch.Tensor, target: torch.Tensor) -> float:
    predicted = mask_probability >= 0.5
    expected = target >= 0.5
    union = (predicted | expected).sum()
    if int(union) == 0:
        return 1.0
    return float((predicted & expected).sum() / union)


def element_metrics(prediction, target):
    mask_probability = prediction["mask_logits"].sigmoid()
    type_probability = prediction["type_logits"].softmax(dim=-1)
    presence = prediction["presence_logits"].sigmoid() >= 0.5
    ious = []
    type_hits = []
    count_errors = []
    mean_probabilities = []
    positive_fractions = []
    logit_stds = []
    pairwise_ious = []
    for batch_index in range(mask_probability.shape[0]):
        probability = mask_probability[batch_index]
        flat = probability.flatten(1)
        intersection = flat @ flat.T
        union = (
            flat.sum(dim=1)[:, None]
            + flat.sum(dim=1)[None]
            - intersection
        )
        pairwise = intersection / union.clamp_min(1.0e-6)
        off_diagonal = ~torch.eye(
            probability.shape[0],
            dtype=torch.bool,
            device=probability.device,
        )
        mean_probabilities.append(float(probability.mean()))
        positive_fractions.append(float((probability >= 0.5).float().mean()))
        logit_stds.append(float(prediction["mask_logits"][batch_index].std()))
        pairwise_ious.append(
            float(pairwise[off_diagonal].mean())
            if probability.shape[0] > 1
            else 0.0
        )
        valid = target.valid[batch_index]
        count = int(valid.sum())
        predicted_count = int(presence[batch_index].sum())
        count_errors.append(abs(predicted_count - count))
        if count == 0:
            continue
        target_masks = target.masks[batch_index, :count]
        target_types = target.types[batch_index, :count]
        if target_masks.shape[-2:] != mask_probability.shape[-2:]:
            target_masks = F.interpolate(
                target_masks[:, None],
                size=mask_probability.shape[-2:],
                mode="nearest",
            )[:, 0]
        predicted_flat = mask_probability[batch_index].flatten(1)
        target_flat = target_masks.flatten(1)
        intersection = predicted_flat @ target_flat.T
        denominator = (
            predicted_flat.sum(dim=1)[:, None]
            + target_flat.sum(dim=1)[None]
            - intersection
        )
        mask_cost = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
        class_cost = -type_probability[
            batch_index,
            :,
            target_types,
        ]
        rows, columns = linear_sum_assignment(
            (mask_cost + class_cost).detach().cpu().numpy()
        )
        for row, column in zip(rows, columns):
            ious.append(
                _dice_to_iou(
                    mask_probability[batch_index, row],
                    target_masks[column],
                )
            )
            type_hits.append(
                int(
                    prediction["type_logits"][batch_index, row].argmax()
                    == target_types[column]
                )
            )
    return {
        "matched_iou": float(np.mean(ious)) if ious else 0.0,
        "type_accuracy": float(np.mean(type_hits)) if type_hits else 0.0,
        "count_mae": float(np.mean(count_errors)) if count_errors else 0.0,
        "predicted_queries": float(presence.sum().item()),
        "mask_mean_probability": float(np.mean(mean_probabilities)),
        "mask_positive_fraction": float(np.mean(positive_fractions)),
        "mask_logit_std": float(np.mean(logit_stds)),
        "mask_pairwise_iou": float(np.mean(pairwise_ious)),
        "target_instances": float(
            sum(int(row.sum()) for row in target.valid)
        ),
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model, taxonomy, _payload = restore_distribution_model(
        args.checkpoint,
        device,
        hierarchy_support_strength=args.hierarchy_support_strength,
    )
    if args.disable_parent_reliability:
        for parameter in model.hub.element.parent_reliability.blend.values():
            parameter.data.zero_()
    full_dataset = CubiCasaFloorplanDataset(
        args.data_root,
        args.split,
        image_size=256,
        cache_root=args.cache_root,
        augment=False,
    )
    stop = min(len(full_dataset), args.offset + args.limit)
    dataset = Subset(full_dataset, range(args.offset, stop))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    family_rows = defaultdict(list)
    family_micro = defaultdict(lambda: [0.0, 0.0])
    element_rows = defaultdict(list)
    attention_rows = defaultdict(list)
    group_ids = model.hub.hierarchy.groups
    for target, footprint, floor_ids, _names in loader:
        target = target.to(device)
        footprint = footprint[:, None].to(device)
        distribution = model.contour_initial_state(
            footprint,
            floor_ids=floor_ids.to(device),
        )
        pixel = distribution["pixel"]
        prediction = pixel["logits"].argmax(dim=1)
        inside = footprint[:, 0].bool()
        for family, class_ids in group_ids.items():
            values = []
            for class_id in class_ids:
                expected = (target == class_id) & inside
                if not bool(expected.any()):
                    continue
                predicted = (prediction == class_id) & inside
                intersection = (predicted & expected).sum().item()
                union = (predicted | expected).sum().item()
                family_micro[family][0] += intersection
                family_micro[family][1] += union
                values.append(intersection / max(1, union))
            family_rows[family].extend(values)
        attention_rows["pixel_routing_entropy"].append(
            float(pixel["routing_entropy"])
        )
        attention_rows["pixel_feedback_expert_fraction"].append(
            float(pixel["feedback_effective_expert_fraction"])
        )
        targets = element_targets(
            target,
            taxonomy,
            model.hub.hierarchy,
            max_queries=model.hub.element.query_counts,
        )
        for family, element_prediction in distribution["elements"].items():
            element_rows[family].append(
                element_metrics(element_prediction, targets[family])
            )
            attention_rows[f"{family}_routing_entropy"].append(
                float(element_prediction["routing_entropy"])
            )
            attention_rows[f"{family}_parent_gate"].append(
                float(element_prediction["parent_gate"].mean())
            )
            for key in (
                "parent_raw_gate",
                "parent_attention_gate",
                "parent_reliability_coverage",
                "parent_reliability_blend",
                "parent_reliability_effect",
            ):
                attention_rows[f"{family}_{key}"].append(
                    float(element_prediction[key].mean())
                )
    report = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "offset": args.offset,
        "cases": len(dataset),
        "condition_mode": "contour_only",
        "parent_reliability_disabled": args.disable_parent_reliability,
        "hierarchy_support_strength": model.hierarchy_support_strength,
        "pixel_family": {
            family: {
                "macro_iou": float(np.mean(values)) if values else 0.0,
                "micro_iou": family_micro[family][0]
                / max(1.0, family_micro[family][1]),
                "class_samples": len(values),
            }
            for family, values in family_rows.items()
        },
        "element_family": {
            family: {
                key: float(np.mean([row[key] for row in rows]))
                for key in rows[0]
            }
            for family, rows in element_rows.items()
        },
        "attention": {
            key: float(np.mean(values))
            for key, values in attention_rows.items()
        },
        "cfm_element_blend": float(
            distribution["cfm"]["element_blend"]
        ),
        "best_of_n_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
