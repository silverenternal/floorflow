#!/usr/bin/env python3
"""Render and audit a fixed style-conditioned design example set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.distribution.checkpoint import restore_distribution_model
from floorflow.distribution.sampling import sample_distribution_floorplans
from floorflow.evaluation.visualize import save_comparison_panel
from floorflow.heads.room_program import gate_room_labels_to_program
from floorflow.representation.core import room_area_distribution


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/train/distribution_style_prompt_r94_v2/best.pt"),
    )
    parser.add_argument(
        "--style-bank",
        type=Path,
        default=Path("outputs/style_banks/cubicasa_train_v1.json"),
    )
    parser.add_argument("--style-id", type=int, default=0)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument(
        "--save-images",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--noise", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/eval/style_design_v1"),
    )
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    model, taxonomy, _payload = restore_distribution_model(
        args.checkpoint,
        device,
    )
    dataset = CubiCasaFloorplanDataset(
        ".data/cubicasa5k",
        args.split,
        image_size=256,
        cache_root="outputs/cache/cubicasa5k_full_v2",
        augment=False,
    )
    room_ids = torch.tensor(taxonomy.room_ids, device=device)
    style_ids = torch.tensor([args.style_id], device=device, dtype=torch.long)
    rows = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index in range(min(args.limit, len(dataset))):
        target, footprint, floor_id, name = dataset[index]
        target = target[None].to(device)
        footprint = footprint[None, None].to(device)
        state, _ = sample_distribution_floorplans(
            model,
            footprint,
            floor_ids=torch.tensor([floor_id], device=device),
            style_ids=style_ids,
            steps=args.steps,
            noise_scale=args.noise,
        )
        scores = state[:, room_ids - 1]
        labels = room_ids[scores.argmax(dim=1)]
        program = scores.amax(dim=(2, 3)) > 0.1
        missing = ~program.any(dim=1)
        if missing.any():
            fallback = scores.amax(dim=(2, 3)).argmax(dim=1)
            program[missing, fallback[missing]] = True
        labels = gate_room_labels_to_program(
            labels,
            program,
            taxonomy.room_ids,
        ) * footprint[:, 0].long()
        target_np = target[0].cpu().numpy()
        footprint_np = footprint[0, 0].cpu().numpy().astype(bool)
        labels_np = labels[0].cpu().numpy()
        room_support = np.isin(labels_np, taxonomy.room_ids)
        target_support = np.isin(target_np, taxonomy.room_ids)
        wall_ids = torch.tensor(taxonomy.class_id("wall") - 1, device=device)
        opening_indices = torch.tensor(
            [class_id - 1 for class_id in taxonomy.opening_ids],
            device=device,
        )
        furniture_indices = torch.tensor(
            [class_id - 1 for class_id in taxonomy.furniture_ids],
            device=device,
        )
        wall_support = state[0, wall_ids] >= 0.5
        opening_support = state[0, opening_indices].sum(dim=0) >= 0.2
        furniture_support = state[0, furniture_indices].sum(dim=0) >= 0.2
        room_support_tensor = torch.isin(labels[0], room_ids)
        wall_band = F.max_pool2d(
            wall_support.float()[None, None],
            kernel_size=5,
            stride=1,
            padding=2,
        )[0, 0] > 0.5
        opening_alignment = float(
            (opening_support & wall_band).sum()
            / opening_support.sum().clamp_min(1)
        )
        furniture_containment = float(
            (furniture_support & room_support_tensor).sum()
            / furniture_support.sum().clamp_min(1)
        )
        coverage = float(
            (room_support & footprint_np).sum()
            / max(1, footprint_np.sum())
        )
        area_mae = float(
            np.abs(
                room_area_distribution(labels_np, taxonomy)
                - room_area_distribution(target_np, taxonomy)
            ).mean()
        )
        outside = int(((labels_np != 0) & ~footprint_np).sum())
        if args.save_images:
            save_comparison_panel(
                args.output_dir / "examples" / f"{index:02d}.png",
                footprint_np,
                target_np,
                labels_np,
                title=f"{name} | style={args.style_id}",
                taxonomy=taxonomy,
            )
        rows.append(
            {
                "id": str(name),
                "outside_footprint_pixels": outside,
                "room_coverage": coverage,
                "room_area_distribution_mae": area_mae,
                "opening_wall_alignment": opening_alignment,
                "furniture_room_containment": furniture_containment,
                "predicted_room_pixels": int(room_support.sum()),
                "target_room_pixels": int(target_support.sum()),
            }
        )
    report = {
        "status": "ok" if all(
            row["outside_footprint_pixels"] == 0 for row in rows
        ) else "fail",
        "checkpoint": str(args.checkpoint),
        "style_bank": str(args.style_bank),
        "style_id": args.style_id,
        "cases": len(rows),
        "best_of_n_used": False,
        "target_used_for_generation": False,
        "summary": {
            key: float(np.mean([row[key] for row in rows]))
            for key in (
                "room_coverage",
                "room_area_distribution_mae",
                "opening_wall_alignment",
                "furniture_room_containment",
            )
        },
        "rows": rows,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
