#!/usr/bin/env python3
"""Train the block-token flow model on the real FloorSet dataset21_lite.pth."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from floorflow.chip_flow.model import (
    ChipBlockFlowModel,
    block_features,
    decode_boxes,
    encode_latent,
)
from floorflow.chip_physics import chip_physics_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--physics-weight", type=float, default=0.05)
    parser.add_argument("--token-ot", action="store_true")
    parser.add_argument("--sparse-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_dataset() -> list[dict]:
    raw = torch.load(
        REPO_ROOT / "third_party/Trinity/data/dataset21_lite.pth",
        map_location="cpu",
        weights_only=False,
    )
    cases = []
    for areas, gt, _b2b_simple, b2b, p2b, pins in raw:
        width, height, x, y = gt.unbind(dim=1)
        boxes = torch.stack([x, y, width, height], dim=1).float()
        constraints = torch.zeros(areas.numel(), 5)
        cases.append(
            {
                "areas": areas.float(),
                "boxes": boxes,
                "b2b": b2b.float(),
                "p2b": p2b.float(),
                "pins": pins.float(),
                "constraints": constraints,
                "features": block_features(
                    areas.float(),
                    constraints,
                    pins.float(),
                    b2b.float(),
                ),
            }
        )
    return cases


def shelf_pack(areas: torch.Tensor) -> torch.Tensor:
    width = areas.sum().sqrt() * 1.05
    order = torch.argsort(areas, descending=True)
    boxes = torch.zeros(areas.numel(), 4)
    cursor_x = cursor_y = row_height = 0.0
    for index in order.tolist():
        side = float(areas[index]) ** 0.5
        if cursor_x + side > width and cursor_x > 0:
            cursor_x = 0.0
            cursor_y += row_height
            row_height = 0.0
        boxes[index] = torch.tensor([cursor_x, cursor_y, side, side])
        cursor_x += side
        row_height = max(row_height, side)
    return boxes


def token_ot_reorder(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    cost = torch.cdist(source, target).pow(2).detach().cpu().numpy()
    rows, columns = linear_sum_assignment(cost)
    order = torch.empty(target.shape[0], dtype=torch.long)
    order[torch.as_tensor(rows)] = torch.as_tensor(columns)
    return target[order]


@torch.no_grad()
def evaluate(cases: list[dict], model: ChipBlockFlowModel) -> dict[str, float]:
    model.eval()
    rows = []
    for case in cases:
        areas = case["areas"]
        latent = encode_latent(shelf_pack(areas), areas)
        for step in range(8):
            time = torch.full((1,), step / 8.0)
            latent = latent + model(
                latent[None], time, case["features"][None]
            )[0] / 8.0
        boxes = decode_boxes(latent, areas)
        terms = chip_physics_loss(
            boxes,
            areas,
            case["constraints"],
            case["b2b"],
            case["p2b"],
            case["pins"],
        )
        target = encode_latent(case["boxes"], areas)
        rows.append(
            {
                "latent_mse": float((latent - target).pow(2).mean()),
                **{name: float(value) for name, value in terms.items()},
            }
        )
    return {
        name: float(np.mean([row[name] for row in rows]))
        for name in rows[0]
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cases = load_dataset()
    order = list(range(len(cases)))
    random.Random(args.seed).shuffle(order)
    split = len(cases) - 408
    train = [cases[index] for index in order[:split]]
    val = [cases[index] for index in order[split:]]
    model = ChipBlockFlowModel(top_k=args.sparse_k or None)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1.0e-4
    )
    history = []
    for step in range(args.steps):
        batch = random.sample(train, args.batch_size)
        latents_target = []
        latents_source = []
        features = []
        physics = []
        for case in batch:
            target = encode_latent(case["boxes"], case["areas"])
            source = encode_latent(shelf_pack(case["areas"]), case["areas"])
            if args.token_ot:
                target = token_ot_reorder(source, target)
            latents_source.append(source)
            latents_target.append(target)
            features.append(case["features"])
        x0 = torch.stack(latents_source) + 0.05 * torch.randn(
            len(batch), 21, 3
        )
        x1 = torch.stack(latents_target)
        features_t = torch.stack(features)
        time = torch.rand(len(batch), 1)
        xt = (1.0 - time[:, :, None]) * x0 + time[:, :, None] * x1
        velocity = model(xt, time[:, 0], features_t)
        flow = (velocity - (x1 - x0)).pow(2).mean()
        predicted = xt + (1.0 - time[:, :, None]) * velocity
        for index, case in enumerate(batch):
            terms = chip_physics_loss(
                decode_boxes(predicted[index], case["areas"]),
                case["areas"],
                case["constraints"],
                case["b2b"],
                case["p2b"],
                case["pins"],
            )
            physics.append(terms["total"])
        physics_loss = torch.stack(physics).mean()
        loss = flow + float(args.physics_weight) * physics_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 100 == 0 or step == args.steps - 1:
            row = {
                "step": step,
                "train_loss": float(loss.detach()),
                "flow": float(flow.detach()),
                "physics": float(physics_loss.detach()),
            }
            row.update({f"val_{k}": v for k, v in evaluate(val, model).items()})
            history.append(row)
            print(json.dumps(row), flush=True)
    report = {
        "config": vars(args) | {"output": str(args.output)},
        "train_cases": len(train),
        "val_cases": len(val),
        "final": evaluate(val, model),
        "history": history,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
