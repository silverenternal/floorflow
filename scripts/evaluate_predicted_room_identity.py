#!/usr/bin/env python3
"""Screen room-identity consolidation on the strongest target-free stack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.distribution.checkpoint import restore_distribution_model
from floorflow.evaluation.paired_bootstrap import paired_delta
from floorflow.evaluation.room_metric_suite import room_metric_suite
from floorflow.heads.room_mask_decoder import RoomMaskDecoder
from floorflow.heads.room_mask_fusion import fused_room_mask_logits
from floorflow.inference.contour_program import (
    load_room_program_predictor,
    predict_room_program,
)
from floorflow.inference.distribution_overlay import DistributionOverlaySettings
from floorflow.inference.instance_semantics import (
    consolidate_room_instances,
    enforce_room_instance_count,
    merge_room_instances_to_count,
)
from floorflow.inference.overlay_decoder import decode_overlay
from floorflow.inference.room_repair import (
    close_room_partition,
    merge_room_fragments,
)
from floorflow.inference.room_segmenter_prior import (
    load_room_segmenter_prior,
    predict_room_segmenter_prior,
)
from floorflow.representation.core import extract_room_instances
from floorflow.training.room_mask_finetune import load_room_mask_decoder_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/train/distribution_dmd_adversarial_datamap4_v11/best.pt"),
    )
    parser.add_argument(
        "--room-segmenter-checkpoint",
        type=Path,
        default=Path(
            "outputs/train/room_segmenter_r18_uniform_full40_v2/best.pt"
        ),
    )
    parser.add_argument(
        "--program-checkpoint",
        type=Path,
        default=Path("outputs/train/room_program_set_full_v3/best.pt"),
    )
    parser.add_argument(
        "--room-mask-checkpoint",
        type=Path,
        default=Path("outputs/train/room_mask_decoder_curriculum_1024_v1/best.pt"),
    )
    parser.add_argument("--data-root", type=Path, default=Path(".data/cubicasa5k"))
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("outputs/cache/cubicasa5k_full_v2"),
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=491)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--room-prior-weight", type=float, default=2.0)
    parser.add_argument("--mask-prior-weight", type=float, default=0.25)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/reports/predicted_room_identity_screen_v1.json"
        ),
    )
    return parser.parse_args()


def load_room_mask(path: Path, in_channels: int, taxonomy, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = payload.get("config", {})
    decoder = RoomMaskDecoder(
        in_channels,
        len(taxonomy.room_ids),
        hidden_dim=int(config.get("hidden_dim", 128)),
        num_queries=int(config.get("queries", 32)),
        layers=int(config.get("layers", 3)),
        heads=int(config.get("heads", 4)),
    ).to(device)
    load_room_mask_decoder_state(decoder, payload["head"])
    return decoder.eval()


def count_instances(labels: torch.Tensor, taxonomy) -> int:
    array = labels.detach().cpu().numpy()
    if array.ndim == 3:
        array = array[0]
    return len(
        extract_room_instances(
            array,
            taxonomy,
            min_area=8,
        )
    )


def metric_row(
    labels: torch.Tensor,
    target: torch.Tensor,
    footprint: torch.Tensor,
    taxonomy,
    *,
    predicted_count: int,
    target_count: int,
    program_count: int,
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
        "predicted_instances": int(predicted_count),
        "target_instances": int(target_count),
        "program_count": int(program_count),
        "instance_count_abs_error": int(abs(predicted_count - target_count)),
    }


def decode_prior(
    model,
    prior: torch.Tensor,
    footprint: torch.Tensor,
    taxonomy,
    floor_ids: torch.Tensor,
    style_ids: torch.Tensor,
    room_prior_weight: float,
    settings: DistributionOverlaySettings,
):
    return decode_overlay(
        model,
        None,
        None,
        footprint,
        taxonomy,
        floor_ids=floor_ids,
        style_ids=style_ids,
        room_prior_logits=prior,
        room_prior_weight=room_prior_weight,
        room_prior_confidence_threshold=1.0,
        settings=settings,
    )


def merge_to_count(
    labels: torch.Tensor,
    logits: torch.Tensor,
    footprint: torch.Tensor,
    room_ids: tuple[int, ...],
    count: int,
) -> torch.Tensor:
    return merge_room_instances_to_count(
        labels,
        logits,
        footprint,
        room_ids,
        int(count),
        min_area=8,
    )


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model, taxonomy, _payload = restore_distribution_model(
        args.checkpoint,
        device,
    )
    segmenter, _segmenter_config = load_room_segmenter_prior(
        args.room_segmenter_checkpoint,
        taxonomy,
        device,
    )
    program_backbone, program_head = load_room_program_predictor(
        args.program_checkpoint,
        taxonomy,
        device,
    )
    room_mask = load_room_mask(
        args.room_mask_checkpoint,
        model.base.output.in_channels,
        taxonomy,
        device,
    )
    full_dataset = CubiCasaFloorplanDataset(
        args.data_root,
        args.split,
        image_size=256,
        cache_root=args.cache_root,
        augment=False,
    )
    stop = min(len(full_dataset), args.offset + args.limit)
    loader = DataLoader(
        Subset(full_dataset, range(args.offset, stop)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    settings = DistributionOverlaySettings(opening_element_weight=0.0)
    room_ids = tuple(model.hub.hierarchy.room_ids)
    variant_names = (
        "segmenter",
        "segmenter_semantic_vote",
        "segmenter_pred_count",
        "segmenter_pred_count_vote",
        "segmenter_merge_pred_count",
        "segmenter_merge_pred_count_plus_1",
        "segmenter_merge_pred_count_plus_2",
        "segmenter_merge_fragments",
        "segmenter_merge_16",
        "segmenter_merge_32",
        "segmenter_merge_64",
        "segmenter_close_support",
        "segmenter_mask_prior",
        "segmenter_mask_prior_merge_32",
        "segmenter_mask_prior_merge_pred_count",
        "oracle_merge_count_diagnostic",
        "oracle_count_diagnostic",
    )
    rows = {name: [] for name in variant_names}
    for target, footprint_raw, floor_ids, names in loader:
        target = target.to(device)
        footprint = footprint_raw[:, None].to(device)
        floor_ids = floor_ids.to(device)
        style_ids = torch.zeros(
            footprint.shape[0],
            dtype=torch.long,
            device=device,
        )
        segmenter_logits = predict_room_segmenter_prior(
            segmenter,
            footprint,
        )
        features = model._encode_features(
            torch.zeros(
                footprint.shape[0],
                model.target_channels,
                *footprint.shape[-2:],
                device=device,
            ),
            torch.zeros(footprint.shape[0], device=device),
            footprint,
            floor_ids=floor_ids,
        )
        mask_logits = fused_room_mask_logits(
            room_mask(features, prototype_logits=segmenter_logits[:, None]),
            len(taxonomy.room_ids),
        )
        segmenter_log_probability = segmenter_logits.log_softmax(dim=1)
        mask_log_probability = mask_logits.log_softmax(dim=1)
        blended_prior = (
            (1.0 - args.mask_prior_weight) * segmenter_log_probability
            + args.mask_prior_weight * mask_log_probability
        )

        segmenter_decoded = decode_prior(
            model,
            segmenter_logits,
            footprint,
            taxonomy,
            floor_ids,
            style_ids,
            args.room_prior_weight,
            settings,
        )
        mask_decoded = decode_prior(
            model,
            blended_prior,
            footprint,
            taxonomy,
            floor_ids,
            style_ids,
            args.room_prior_weight,
            settings,
        )
        program = predict_room_program(
            program_backbone,
            program_head,
            footprint,
            floor_ids,
        )
        predicted_counts = program.sum(dim=1).clamp_min(1)
        conditioned_room_logits = segmenter_decoded["conditioned_logits"][
            :, room_ids
        ]
        base_rooms = segmenter_decoded["labels"]
        semantic_rooms = consolidate_room_instances(
            base_rooms,
            conditioned_room_logits,
            footprint,
            room_ids,
        )
        for index, name in enumerate(names):
            masks = segmenter_decoded["labels"][index : index + 1]
            target_count = count_instances(target[index], taxonomy)
            count_rooms = enforce_room_instance_count(
                masks,
                conditioned_room_logits[index : index + 1],
                footprint[index : index + 1],
                room_ids,
                int(predicted_counts[index]),
            )
            count_vote_rooms = consolidate_room_instances(
                count_rooms,
                conditioned_room_logits[index : index + 1],
                footprint[index : index + 1],
                room_ids,
            )
            oracle_rooms = enforce_room_instance_count(
                masks,
                conditioned_room_logits[index : index + 1],
                footprint[index : index + 1],
                room_ids,
                target_count,
            )
            merged_fragments = torch.as_tensor(
                merge_room_fragments(
                    masks[0].cpu().numpy(),
                    taxonomy,
                ),
                device=device,
                dtype=torch.long,
            )[None]
            merged_16 = torch.as_tensor(
                merge_room_fragments(
                    masks[0].cpu().numpy(),
                    taxonomy,
                    min_area=16,
                ),
                device=device,
                dtype=torch.long,
            )[None]
            merged_32 = torch.as_tensor(
                merge_room_fragments(
                    masks[0].cpu().numpy(),
                    taxonomy,
                    min_area=32,
                ),
                device=device,
                dtype=torch.long,
            )[None]
            merged_64 = torch.as_tensor(
                merge_room_fragments(
                    masks[0].cpu().numpy(),
                    taxonomy,
                    min_area=64,
                ),
                device=device,
                dtype=torch.long,
            )[None]
            mask_prior_rooms = mask_decoded["labels"][
                index : index + 1
            ]
            mask_prior_merge_32 = torch.as_tensor(
                merge_room_fragments(
                    mask_prior_rooms[0].cpu().numpy(),
                    taxonomy,
                    min_area=32,
                ),
                device=device,
                dtype=torch.long,
            )[None]
            merge_pred = merge_to_count(
                masks,
                conditioned_room_logits[index : index + 1],
                footprint[index : index + 1],
                room_ids,
                int(predicted_counts[index]),
            )
            merge_pred_plus_1 = merge_to_count(
                masks,
                conditioned_room_logits[index : index + 1],
                footprint[index : index + 1],
                room_ids,
                int(predicted_counts[index]) + 1,
            )
            merge_pred_plus_2 = merge_to_count(
                masks,
                conditioned_room_logits[index : index + 1],
                footprint[index : index + 1],
                room_ids,
                int(predicted_counts[index]) + 2,
            )
            merge_oracle = merge_to_count(
                masks,
                conditioned_room_logits[index : index + 1],
                footprint[index : index + 1],
                room_ids,
                target_count,
            )
            mask_prior_merge_pred = merge_to_count(
                mask_prior_rooms,
                mask_decoded["conditioned_logits"][
                    index : index + 1, room_ids
                ],
                footprint[index : index + 1],
                room_ids,
                int(predicted_counts[index]),
            )
            closed_support = torch.as_tensor(
                close_room_partition(
                    masks[0].cpu().numpy(),
                    footprint[index, 0].cpu().numpy(),
                    taxonomy,
                ),
                device=device,
                dtype=torch.long,
            )[None]
            variants = {
                "segmenter": masks,
                "segmenter_semantic_vote": semantic_rooms[index : index + 1],
                "segmenter_pred_count": count_rooms,
                "segmenter_pred_count_vote": count_vote_rooms,
                "segmenter_merge_fragments": merged_fragments,
                "segmenter_merge_16": merged_16,
                "segmenter_merge_32": merged_32,
                "segmenter_merge_64": merged_64,
                "segmenter_close_support": closed_support,
                "segmenter_mask_prior": mask_prior_rooms,
                "segmenter_mask_prior_merge_32": mask_prior_merge_32,
                "segmenter_mask_prior_merge_pred_count": mask_prior_merge_pred,
                "segmenter_merge_pred_count": merge_pred,
                "segmenter_merge_pred_count_plus_1": merge_pred_plus_1,
                "segmenter_merge_pred_count_plus_2": merge_pred_plus_2,
                "oracle_merge_count_diagnostic": merge_oracle,
                "oracle_count_diagnostic": oracle_rooms,
            }
            for variant, prediction in variants.items():
                predicted_count = count_instances(prediction, taxonomy)
                rows[variant].append(
                    {
                        "case": name,
                        **metric_row(
                            prediction[0],
                            target[index],
                            footprint[index, 0],
                            taxonomy,
                            predicted_count=predicted_count,
                            target_count=target_count,
                            program_count=int(program[index].sum()),
                        ),
                    }
                )
    metrics = {
        name: {
            key: float(np.mean([row[key] for row in values]))
            for key in values[0]
            if key != "case"
        }
        for name, values in rows.items()
    }
    baseline = rows["segmenter"]
    paired = {
        name: {
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
                "instance_count_abs_error",
            )
        }
        for name, values in rows.items()
        if name != "segmenter"
    }
    report = {
        "schema_version": "predicted-room-identity-screen-v1",
        "scope": "target-free val screen; test labels not read",
        "split": args.split,
        "cases": stop - args.offset,
        "checkpoint": str(args.checkpoint),
        "room_segmenter_checkpoint": str(args.room_segmenter_checkpoint),
        "program_checkpoint": str(args.program_checkpoint),
        "room_mask_checkpoint": str(args.room_mask_checkpoint),
        "room_prior_weight": args.room_prior_weight,
        "mask_prior_weight": args.mask_prior_weight,
        "metrics": metrics,
        "paired": paired,
        "oracle_count_diagnostic": (
            "Uses target instance count only as a non-deployable ceiling."
        ),
        "next_decision": (
            "Train a new identity head only if a deployable count or prior "
            "arm improves room identity without support collapse."
        ),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "cases": report["cases"],
                "metrics": metrics,
            },
            indent=2,
        )
    )


__all__ = ["main"]


if __name__ == "__main__":
    main()
