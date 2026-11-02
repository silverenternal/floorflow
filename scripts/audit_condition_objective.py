#!/usr/bin/env python3
"""One-batch audit of the legacy and deployment-aligned condition objectives."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import torch
import torch.nn.functional as F

from floorflow.data.class_balance import load_positive_weights
from floorflow.objectives.condition_objective import ConditionObjectiveWeights, condition_layout_loss
from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.objectives.loss_audit import ObjectiveTerm, contribution_rows, gradient_contribution_rows
from floorflow.objectives.semantic_losses import boundary_focal_dice, room_group_loss, wall_focal_dice
from floorflow.training.checkpoint import load_checkpoint_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--partition-checkpoint", type=Path, default=None)
    parser.add_argument("--object-checkpoint", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / ".data/cubicasa5k")
    parser.add_argument("--cache-root", type=Path, default=REPO_ROOT / "outputs/cache/cubicasa5k_full_v2")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs/diagnostics/condition_objective")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dataset = CubiCasaFloorplanDataset(
        args.data_root,
        "train",
        image_size=args.image_size,
        cache_root=args.cache_root,
        augment=False,
    )
    taxonomy = dataset.taxonomy
    model, _config, checkpoint_taxonomy, _state = load_checkpoint_model(
        args.checkpoint,
        device,
        taxonomy=taxonomy,
    )
    taxonomy = checkpoint_taxonomy
    model.train()
    tensors = [dataset[index] for index in range(min(args.batch_size, len(dataset)))]
    target = torch.stack([item[0] for item in tensors]).to(device)
    footprint = torch.stack([item[1] for item in tensors]).to(device)
    floor_ids = torch.tensor([item[2] for item in tensors], device=device)
    zero = torch.zeros(target.shape[0], model.target_channels, *target.shape[-2:], device=device)
    _, aux = model.forward_with_aux(zero, torch.zeros(target.shape[0], device=device), footprint, floor_ids=floor_ids)

    positive_np = load_positive_weights(args.cache_root, taxonomy.num_target_channels, 20.0)
    positive_weights = torch.from_numpy(positive_np).to(device)
    room_weights = positive_weights[
        torch.tensor([class_id - 1 for class_id in taxonomy.room_ids], device=device)
    ]
    room_weights = room_weights / room_weights.mean()
    new_loss, new_components, new_terms = condition_layout_loss(
        aux,
        target,
        footprint,
        taxonomy,
        room_class_weights=room_weights,
        positive_class_weights=positive_weights,
        weights=ConditionObjectiveWeights(),
    )

    semantic_logits = aux["semantic_logits"].float()
    categorical = torch.cat([torch.zeros_like(semantic_logits[:, :1]), semantic_logits], dim=1)
    class_weights = torch.cat([torch.ones(1, device=device), positive_weights])
    full_ce = F.cross_entropy(categorical, target, weight=class_weights)
    foreground = (target > 0).unsqueeze(1).float()
    legacy_boundary = boundary_focal_dice(
        semantic_logits,
        F.one_hot(target.clamp_min(1) - 1, taxonomy.num_target_channels)
        .permute(0, 3, 1, 2)
        .float()
        * foreground,
    )
    wall_id = taxonomy.class_id("wall")
    legacy_wall = wall_focal_dice(semantic_logits[:, wall_id - 1], (target == wall_id).float())
    room_loss, _room_ce, _room_dice = room_group_loss(
        semantic_logits,
        target,
        taxonomy,
        room_weights,
    )
    legacy_terms = [
        ObjectiveTerm("legacy_full_ce", full_ce, 1.0),
        ObjectiveTerm("legacy_all_foreground_boundary", legacy_boundary, 0.10),
        ObjectiveTerm("legacy_wall", legacy_wall, 0.10),
        ObjectiveTerm("legacy_visible_room_ce_dice", room_loss, 0.50),
    ]
    new_rows = contribution_rows(new_terms)
    legacy_rows = contribution_rows(legacy_terms)
    new_gradients = gradient_contribution_rows(new_terms, model.parameters(), primary="room_partition")
    legacy_gradients = gradient_contribution_rows(
        legacy_terms,
        model.parameters(),
        primary="legacy_full_ce",
    )
    payload = {
        "batch_size": int(target.shape[0]),
        "new_total": float(new_loss.detach().cpu()),
        "new_components": {key: float(value.detach().cpu()) for key, value in new_components.items()},
        "new_contributions": new_rows,
        "new_gradient_contributions": new_gradients,
        "legacy_total": float(sum(term.weighted_value for term in legacy_terms).detach().cpu()),
        "legacy_contributions": legacy_rows,
        "legacy_gradient_contributions": legacy_gradients,
    }
    (args.output_dir / "objective_audit.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"new": new_rows, "legacy": legacy_rows}, indent=2))


if __name__ == "__main__":
    main()
