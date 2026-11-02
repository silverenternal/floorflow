#!/usr/bin/env python3
"""Audit learned mask coverage, attribute retention, and MaskPLAN overlap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.flow.condition_view import sparse_control_view
from floorflow.flow.guided_control import guided_source_from_labels
from floorflow.flow.mask_policies import AttributeMaskPolicy
from floorflow.flow.masked_control import control_channel_groups
from floorflow.model.learned_mask_generator import LearnedMaskGenerator
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
    parser.add_argument("--ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=53)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "mask_generator" not in payload:
        raise ValueError("checkpoint has no learned mask generator")
    model, config, taxonomy, _state = load_checkpoint_model(
        args.checkpoint,
        args.device,
        taxonomy=None,
    )
    generator = LearnedMaskGenerator(model.target_channels).to(args.device)
    generator.load_state_dict(payload["mask_generator"])
    generator.eval()
    dataset = CubiCasaFloorplanDataset(
        args.data_root,
        args.split,
        image_size=int(config.get("image_size", 256)),
        cache_root=args.cache_root,
        limit=args.limit,
        augment=False,
    )
    groups = {
        "room": torch.tensor(taxonomy.room_ids, device=args.device) - 1,
        "wall": torch.tensor(
            [taxonomy.class_id("wall") - 1],
            device=args.device,
        ),
        "opening": torch.tensor(
            taxonomy.opening_ids,
            device=args.device,
        ) - 1,
        "furniture": torch.tensor(
            taxonomy.furniture_ids,
            device=args.device,
        ) - 1,
    }
    coverage = []
    overlaps = []
    retention = {name: [] for name in groups}
    for index in range(len(dataset)):
        target, footprint, _floor_id, _name = dataset[index]
        target = target[None].to(args.device)
        footprint = footprint[None].to(args.device)
        source = guided_source_from_labels(target, footprint, taxonomy)
        seed = AttributeMaskPolicy().sample(
            source,
            footprint,
            taxonomy,
            ratio=min(0.75, args.ratio * 0.5),
        )
        view = sparse_control_view(source, footprint, seed.value)
        groups_by_name = control_channel_groups(taxonomy)
        generated = generator(
            view.control,
            view.footprint,
            target_ratio=args.ratio,
            known_mask=seed.value,
            visibility=view.visibility,
            channel_groups=tuple(
                groups_by_name[name].to(args.device)
                for name in ("room", "wall", "opening", "furniture")
            ),
        )
        torch.manual_seed(args.seed + index)
        attribute = AttributeMaskPolicy().sample(
            source,
            footprint,
            taxonomy,
            ratio=args.ratio,
        )
        union = (generated.hard | attribute.value).sum().clamp_min(1)
        overlaps.append(
            float((generated.hard & attribute.value).sum().float() / union)
        )
        area = footprint[:, None].expand_as(generated.hard).bool()
        coverage.append(
            float(generated.hard[area].float().mean())
        )
        for name, channel_ids in groups.items():
            selected = generated.hard[:, channel_ids]
            selected_area = footprint[:, None].expand_as(selected).bool()
            retention[name].append(
                1.0 - float(selected[selected_area].float().mean())
            )
    report = {
        "checkpoint": str(args.checkpoint),
        "cases": len(dataset),
        "target_mask_ratio": args.ratio,
        "coverage": {
            "mean": float(np.mean(coverage)),
            "std": float(np.std(coverage, ddof=1)),
        },
        "attribute_retention": {
            name: {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)),
            }
            for name, values in retention.items()
        },
        "maskplan_attribute_overlap": {
            "mean": float(np.mean(overlaps)),
            "std": float(np.std(overlaps, ddof=1)),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
