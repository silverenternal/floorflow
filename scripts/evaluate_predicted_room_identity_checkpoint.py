#!/usr/bin/env python3
"""Evaluate a trained predicted-control room-identity checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from train_predicted_room_identity import identity_features

from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.distribution.checkpoint import restore_distribution_model
from floorflow.evaluation.paired_bootstrap import paired_delta
from floorflow.evaluation.room_metric_suite import room_metric_suite
from floorflow.heads.predicted_room_identity import PredictedRoomIdentityHead
from floorflow.heads.room_mask_decoder import RoomMaskDecoder
from floorflow.heads.room_mask_fusion import (
    fused_room_mask_logits,
    mask_region_class_logits,
)
from floorflow.inference.contour_program import (
    load_room_program_predictor,
    predict_room_program,
)
from floorflow.inference.distribution_overlay import DistributionOverlaySettings
from floorflow.inference.overlay_decoder import decode_overlay
from floorflow.inference.room_segmenter_prior import load_room_segmenter_prior
from floorflow.representation.core import extract_room_instances

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--identity-checkpoint",
        type=Path,
        default=ROOT
        / "outputs/train/predicted_room_identity_1024_v1/best.pt",
    )
    parser.add_argument(
        "--distribution-checkpoint",
        type=Path,
        default=ROOT
        / "outputs/train/distribution_dmd_adversarial_datamap4_v11/best.pt",
    )
    parser.add_argument(
        "--room-segmenter-checkpoint",
        type=Path,
        default=ROOT
        / "outputs/train/room_segmenter_r18_uniform_full40_v2/best.pt",
    )
    parser.add_argument("--data-root", type=Path, default=ROOT / ".data/cubicasa5k")
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=ROOT / "outputs/cache/cubicasa5k_full_v2",
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=491)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--blend-weights",
        nargs="+",
        type=float,
        default=(0.0, 0.25, 0.5, 0.75, 1.0),
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--room-prior-weight", type=float, default=2.0)
    parser.add_argument("--program-prior-scale", type=float, default=2.0)
    parser.add_argument(
        "--program-checkpoint",
        type=Path,
        default=ROOT / "outputs/train/room_program_set_full_v3/best.pt",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "outputs/reports/predicted_room_identity_checkpoint_val491_v1.json",
    )
    return parser.parse_args()


def count_instances(labels: torch.Tensor, taxonomy) -> int:
    array = labels.detach().cpu().numpy()
    if array.ndim == 3:
        array = array[0]
    return len(extract_room_instances(array, taxonomy, min_area=8))


def metric_row(
    labels: torch.Tensor,
    target: torch.Tensor,
    footprint: torch.Tensor,
    taxonomy,
) -> dict[str, float | int]:
    room_ids = torch.tensor(taxonomy.room_ids, device=labels.device)
    support = torch.isin(labels, room_ids)
    target_support = torch.isin(target, room_ids)
    suite = room_metric_suite(
        labels.cpu().numpy(),
        target.cpu().numpy(),
        footprint.cpu().numpy(),
        taxonomy,
        predicted_room_support=support.cpu().numpy(),
        predicted_instance_support=support.cpu().numpy(),
        target_room_support=target_support.cpu().numpy(),
    )
    instance = suite["matched_instance"]
    return {
        "room_macro_iou": float(
            suite["visible_semantic"]["room_macro_iou"]
        ),
        "instance_geometric_iou": float(
            instance["matched_geometric_iou"]
        ),
        "instance_room_f1": float(instance["instance_f1_at_threshold"]),
        "instance_type_accuracy": float(instance["matched_type_accuracy"]),
        "class_constrained_iou": float(
            instance["matched_class_constrained_iou"]
        ),
        "room_support_iou": float(
            suite["occlusion_aware_support"]["room_support_iou"]
        ),
        "predicted_instances": count_instances(labels, taxonomy),
        "target_instances": count_instances(target, taxonomy),
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    distribution, taxonomy, _ = restore_distribution_model(
        args.distribution_checkpoint,
        device,
    )
    segmenter, _ = load_room_segmenter_prior(
        args.room_segmenter_checkpoint,
        taxonomy,
        device,
    )
    program_backbone, program_head = load_room_program_predictor(
        args.program_checkpoint,
        taxonomy,
        device,
    )
    payload = torch.load(
        args.identity_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    config = payload["config"]
    dense_head = "input.0.weight" in payload["model"]
    if dense_head:
        decoder = PredictedRoomIdentityHead(
            int(config["feature_channels"]),
            len(taxonomy.room_ids),
            hidden_channels=int(config["hidden_channels"]),
            blocks=int(config["blocks"]),
            use_image=bool(config.get("use_image", False)),
        ).to(device)
    else:
        decoder = RoomMaskDecoder(
            int(config["feature_channels"]),
            len(taxonomy.room_ids),
            hidden_dim=int(config["hidden_dim"]),
            num_queries=int(config["queries"]),
            layers=int(config["layers"]),
            heads=int(config["heads"]),
        ).to(device)
    decoder.load_state_dict(payload["model"])
    decoder.eval()
    dataset = CubiCasaFloorplanDataset(
        args.data_root,
        args.split,
        image_size=256,
        cache_root=args.cache_root,
        augment=False,
        include_image=bool(config.get("use_image", False)),
    )
    stop = min(len(dataset), args.offset + args.limit)
    loader = DataLoader(
        Subset(dataset, range(args.offset, stop)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    room_ids = tuple(distribution.hub.hierarchy.room_ids)
    settings = DistributionOverlaySettings(opening_element_weight=0.0)
    style_id = torch.zeros(1, dtype=torch.long, device=device)
    weights = tuple(dict.fromkeys(float(value) for value in args.blend_weights))
    for value in weights:
        if not 0.0 <= value <= 1.0:
            raise ValueError("blend weights must be in [0,1]")
    rows = {str(weight): [] for weight in weights}
    if not dense_head:
        rows.update({f"region_{weight}": [] for weight in weights[1:]})
    rows.update({f"program_{weight}": [] for weight in weights[1:]})
    for target, footprint, floor_ids, names, *extra in loader:
        target = target.to(device)
        footprint = footprint.to(device)
        floor_ids = floor_ids.to(device)
        image = extra[0].to(device) if extra else None
        batch_style_ids = style_id.expand(footprint.shape[0])
        features, segmenter_logits = identity_features(
            distribution,
            segmenter,
            footprint,
            floor_ids,
            room_ids,
        )
        predictions = None
        if dense_head:
            identity_logits = decoder(
                features,
                segmenter_logits,
                image=image,
            )["room_logits"]
        else:
            predictions = decoder(
                features,
                prototype_logits=segmenter_logits[:, None],
            )
            identity_logits = fused_room_mask_logits(
                predictions,
                len(taxonomy.room_ids),
            )
        segmenter_probability = segmenter_logits.log_softmax(dim=1)
        identity_probability = identity_logits.log_softmax(dim=1)
        program = predict_room_program(
            program_backbone,
            program_head,
            footprint,
            floor_ids,
        )
        program_logits = (
            identity_logits
            + args.program_prior_scale
            * (program.float() * 2.0 - 1.0)[:, :, None, None]
        )
        program_probability = program_logits.log_softmax(dim=1)
        for index, name in enumerate(names):
            priors = []
            for weight in weights:
                priors.append(
                    (
                        str(weight),
                        (1.0 - weight)
                        * segmenter_probability[index : index + 1]
                        + weight * identity_probability[index : index + 1],
                    )
                )
                if weight != weights[0]:
                    if predictions is not None:
                        region_logits = mask_region_class_logits(
                            {
                                "mask_logits": predictions["mask_logits"][
                                    index : index + 1
                                ]
                            },
                            segmenter_logits[index : index + 1],
                            footprint[index : index + 1],
                        )
                        priors.append(
                            (
                                f"region_{weight}",
                                (1.0 - weight)
                                * segmenter_probability[index : index + 1]
                                + weight * region_logits.log_softmax(dim=1),
                            )
                        )
                    priors.append(
                        (
                            f"program_{weight}",
                            (1.0 - weight)
                            * segmenter_probability[index : index + 1]
                            + weight * program_probability[index : index + 1],
                        )
                    )
            for arm, logits in priors:
                decoded = decode_overlay(
                    distribution,
                    None,
                    None,
                    footprint[index : index + 1, None],
                    taxonomy,
                    floor_ids=floor_ids[index : index + 1],
                    style_ids=batch_style_ids[index : index + 1],
                    room_prior_logits=logits,
                    room_prior_weight=args.room_prior_weight,
                    room_prior_confidence_threshold=1.0,
                    settings=settings,
                )
                labels = decoded["labels"]
                rows[arm].append(
                    {
                        "case": name,
                        **metric_row(
                            labels[0],
                            target[index],
                            footprint[index],
                            taxonomy,
                        ),
                    }
                )
    metrics = {
        str(weight): {
            key: float(np.mean([row[key] for row in values]))
            for key in values[0]
            if key != "case"
        }
        for weight, values in rows.items()
    }
    baseline = rows[str(weights[0])]
    paired = {
        str(weight): {
            key: paired_delta(
                values,
                baseline,
                key,
                iterations=args.bootstrap_iterations,
                seed=2902,
            )
            for key in (
                "room_macro_iou",
                "instance_geometric_iou",
                "instance_room_f1",
                "instance_type_accuracy",
            )
        }
        for weight, values in rows.items()
        if weight != str(weights[0])
    }
    report = {
        "schema_version": "predicted-room-identity-checkpoint-v1",
        "scope": "target-free validation; test labels not read",
        "split": args.split,
        "cases": stop - args.offset,
        "identity_checkpoint": str(args.identity_checkpoint),
        "identity_head": "dense_residual" if dense_head else "query_mask",
        "distribution_checkpoint": str(args.distribution_checkpoint),
        "room_segmenter_checkpoint": str(args.room_segmenter_checkpoint),
        "blend_definition": (
            "(1-w) * log_softmax(segmenter) + "
            "w * log_softmax(identity_masks)"
        ),
        "metrics": metrics,
        "paired_vs_segmenter": paired,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "metrics": metrics}, indent=2))


__all__ = ["main"]


if __name__ == "__main__":
    main()
