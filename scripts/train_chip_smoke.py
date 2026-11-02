#!/usr/bin/env python3
"""Tiny chip-floorplanning flow-matching run on the 100 official FloorSet cases."""

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
sys.path.insert(0, str(REPO_ROOT / "third_party/floorset-src"))

from lite_dataset_test import FloorplanDatasetLiteTest

from floorflow.chip_flow.model import (
    ChipBlockFlowModel,
    block_features,
    decode_boxes,
    encode_latent,
)
from floorflow.chip_physics import chip_physics_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--physics-weight", type=float, default=0.05)
    parser.add_argument("--token-ot", action="store_true")
    parser.add_argument("--sparse-k", type=int, default=16)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_cases() -> list[dict]:
    cache = REPO_ROOT / "outputs/cache/chip_cases_v1.pt"
    if cache.is_file():
        return torch.load(cache, map_location="cpu", weights_only=False)
    dataset = FloorplanDatasetLiteTest(str(REPO_ROOT / "third_party/floorset-src"))
    cases = []
    for index in range(len(dataset)):
        item = dataset[index]
        areas, b2b, p2b, pins, constraints = item["input"]
        polygons = item["label"][0]
        boxes = []
        for poly in polygons:
            valid = poly[(poly >= -0.5).all(dim=1)]
            low = valid.min(dim=0).values
            high = valid.max(dim=0).values
            boxes.append(
                torch.stack(
                    [low[0], low[1], high[0] - low[0], high[1] - low[1]]
                )
            )
        cases.append(
            {
                "areas": areas.float(),
                "b2b": b2b.float(),
                "p2b": p2b.float(),
                "pins": pins.float(),
                "constraints": constraints.float(),
                "boxes": torch.stack(boxes),
                "features": block_features(areas, constraints, pins, b2b),
            }
        )
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cases, cache)
    return cases


def shelf_pack(areas: torch.Tensor) -> torch.Tensor:
    """A deterministic legal-ish source layout used to anchor the flow."""

    count = areas.numel()
    width = areas.sum().sqrt() * 1.05
    order = torch.argsort(areas, descending=True)
    boxes = torch.zeros(count, 4)
    cursor_x = 0.0
    cursor_y = 0.0
    row_height = 0.0
    for index in order.tolist():
        area = float(areas[index])
        side = area**0.5
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
        source = encode_latent(shelf_pack(areas), areas)
        latent = source
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
    cases = load_cases()
    order = list(range(len(cases)))
    random.Random(args.seed).shuffle(order)
    split = int(len(cases) * 0.8)
    train = [cases[index] for index in order[:split]]
    val = [cases[index] for index in order[split:]]
    model = ChipBlockFlowModel(top_k=args.sparse_k or None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1.0e-4)
    history = []
    for step in range(args.steps):
        batch = random.sample(train, min(args.batch_size, len(train)))
        losses = []
        optimizer.zero_grad()
        for case in batch:
            areas = case["areas"]
            x1 = encode_latent(case["boxes"], areas)
            x0 = encode_latent(shelf_pack(areas), areas) + 0.05 * torch.randn_like(x1)
            if args.token_ot:
                x1 = token_ot_reorder(x0, x1)
            time = torch.rand(1)
            xt = (1.0 - time) * x0 + time * x1
            velocity = model(xt[None], time, case["features"][None])[0]
            flow = (velocity - (x1 - x0)).pow(2).mean()
            predicted = xt + (1.0 - time) * velocity
            terms = chip_physics_loss(
                decode_boxes(predicted, areas),
                areas,
                case["constraints"],
                case["b2b"],
                case["p2b"],
                case["pins"],
            )
            losses.append(flow + float(args.physics_weight) * terms["total"])
        loss = torch.stack(losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 25 == 0 or step == args.steps - 1:
            row = {"step": step, "train_loss": float(loss)}
            row.update({f"val_{k}": v for k, v in evaluate(val, model).items()})
            history.append(row)
            print(json.dumps(row), flush=True)
    model.eval()
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
