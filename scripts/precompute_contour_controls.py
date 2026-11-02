#!/usr/bin/env python3
"""Precompute stage-1 contour controls for stage-2 refinement training."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from floorflow.data.control_cache import control_cache_path, write_control_cache
from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.flow.guided_control import guided_source_from_labels
from floorflow.flow.mask_policies import AttributeMaskPolicy, make_mask_policy
from floorflow.inference.contour_controls import (
    controls_from_target_channels,
    predict_contour_controls,
)
from floorflow.model.learned_mask_generator import LearnedMaskGenerator
from floorflow.model.sparse_control_attention import (
    MaskedControlFlow,
    sample_sparse_guided_floorplans,
)
from floorflow.training.checkpoint import load_checkpoint_model
from floorflow.training.masked_conditioning import prepare_masked_condition


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path(".data/cubicasa5k"))
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("outputs/cache/cubicasa5k_full_v2"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/cache/contour_controls_b1"),
    )
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--flow-steps", type=int, default=4)
    parser.add_argument("--masked-control", action="store_true")
    parser.add_argument("--mask-ratio", type=float, default=0.25)
    parser.add_argument(
        "--mask-policy",
        choices=("spatial", "attribute", "composite", "learned"),
        default="learned",
    )
    parser.add_argument(
        "--control-source",
        choices=("aux", "sampled", "blend"),
        default="blend",
    )
    parser.add_argument("--room-size", type=int, default=64)
    parser.add_argument("--structure-size", type=int, default=64)
    parser.add_argument("--furniture-size", type=int, default=96)
    parser.add_argument("--support-size", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, config, taxonomy, _state = load_checkpoint_model(
        args.checkpoint,
        args.device,
    )
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if args.masked_control:
        if "sparse_attention" not in payload:
            raise ValueError("masked precompute requires sparse-control attention")
        model = MaskedControlFlow(model).to(args.device)
        model.sparse_attention.load_state_dict(
            payload["sparse_attention"],
            strict=False,
        )
    model.eval()
    mask_generator = None
    if args.masked_control and args.mask_policy == "learned":
        if "mask_generator" not in payload:
            raise ValueError("learned mask policy requires mask generator")
        mask_generator = LearnedMaskGenerator(model.target_channels).to(args.device)
        mask_generator.load_state_dict(payload["mask_generator"])
        mask_generator.eval()
    missing_token = payload.get(
        "missing_token",
        torch.zeros(model.target_channels),
    ).to(args.device)
    for split in args.splits:
        dataset = CubiCasaFloorplanDataset(
            args.data_root,
            split,
            image_size=int(config.get("image_size", 256)),
            cache_root=args.cache_root,
            limit=args.limit,
            augment=False,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
        )
        completed = 0
        for target, footprint, floor_ids, names in loader:
            footprint = footprint.to(args.device)
            floor_ids = floor_ids.to(args.device)
            if args.masked_control:
                full_source = guided_source_from_labels(
                    target.to(args.device),
                    footprint,
                    taxonomy,
                )
                prepared = prepare_masked_condition(
                    full_source,
                    footprint,
                    taxonomy,
                    missing_token,
                    missing_ratio=args.mask_ratio,
                    floor_ids=floor_ids,
                    mask_generator=mask_generator,
                    mask_policy=(
                        AttributeMaskPolicy()
                        if args.mask_policy == "learned"
                        else make_mask_policy(args.mask_policy)
                    ),
                )
                sampled = sample_sparse_guided_floorplans(
                    model,
                    prepared.view.footprint,
                    prepared.control,
                    prepared.missing_mask,
                    steps=args.flow_steps,
                    floor_ids=floor_ids,
                )
                controls = controls_from_target_channels(
                    sampled,
                    footprint,
                    taxonomy,
                    structure_size=args.structure_size,
                    furniture_size=args.furniture_size,
                    support_size=args.support_size,
                )
            else:
                controls = predict_contour_controls(
                    model,
                    footprint,
                    taxonomy,
                    floor_ids=floor_ids,
                    flow_steps=args.flow_steps,
                    control_source=args.control_source,
                    structure_size=args.structure_size,
                    furniture_size=args.furniture_size,
                    support_size=args.support_size,
                )
            for index, name in enumerate(names):
                write_control_cache(
                    control_cache_path(args.output_root, split, name),
                    room_logits=controls.room_logits[index : index + 1],
                    program=controls.program[index : index + 1],
                    structure=controls.structure[index : index + 1],
                    furniture=controls.furniture[index : index + 1],
                    support=controls.support[index : index + 1],
                    room_size=args.room_size,
                )
            completed += target.shape[0]
            print(f"{split}: {completed}/{len(dataset)}", flush=True)


if __name__ == "__main__":
    main()
