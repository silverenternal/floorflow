#!/usr/bin/env python3
"""CPU gate for the Trinity token-flow bridge on a real FloorSet batch."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from floorflow.chip_flow.cache import PackedCaseStore
from floorflow.chip_flow.flow_adapter import (
    build_chip_model,
    decode_for_batch,
    encode_target_and_condition,
)
from floorflow.chip_flow.model import decode_boxes_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache",
        type=Path,
        default=ROOT / "outputs/cache/chip_floorset_50k_train_12k_dev_v2/train",
    )
    parser.add_argument("--block-count", type=int, default=31)
    parser.add_argument("--cases", type=int, default=2)
    parser.add_argument(
        "--parity-report",
        type=Path,
        default=ROOT / "outputs/reports/chip_trinity_parity_gap_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_trinity_token_bridge_cpu_v1.json",
    )
    return parser.parse_args()


def model_config() -> dict:
    return {
        "architecture": "trinity_token",
        "feature_dim": 19,
        "hidden": 192,
        "depth": 12,
        "heads": 3,
        "head_dim": 64,
        "graph_pe_k": 2,
    }


def permuted_batch(batch: dict, permutation: torch.Tensor) -> dict:
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(len(permutation))
    result = copy.deepcopy(batch)
    result["b2b"] = torch.cat(
        [
            inverse[batch["b2b"][..., :2].long()].float(),
            batch["b2b"][..., 2:],
        ],
        dim=-1,
    )
    result["p2b"] = batch["p2b"].clone()
    result["p2b"][..., 1] = inverse[batch["p2b"][..., 1].long()].float()
    for key in (
        "areas",
        "boxes",
        "constraints",
        "features",
        "source_latent",
        "target_latent",
    ):
        result[key] = batch[key][:, permutation]
    return result


def main() -> None:
    args = parse_args()
    torch.manual_seed(19)
    store = PackedCaseStore(args.cache)
    batch = store.batch(store.indices(args.block_count)[: args.cases], device="cpu")
    model = build_chip_model(model_config())
    model.train()
    target, condition, context = encode_target_and_condition(model, batch)
    source = torch.randn_like(target)
    time_value = torch.full((target.shape[0],), 0.5)
    started = time.perf_counter()
    prediction = model(source, time_value, condition)
    decoded, _ = decode_for_batch(
        model,
        prediction,
        condition,
        context,
        batch,
    )
    (decoded - 1.0).square().mean().backward()
    backward_seconds = time.perf_counter() - started
    boxes = decode_boxes_batch(decoded.detach(), batch["areas"])
    decoded_areas = boxes[..., 2] * boxes[..., 3]
    area_error = (decoded_areas - batch["areas"]).abs().max()

    parity = json.loads(args.parity_report.read_text(encoding="utf-8"))
    parameters = sum(parameter.numel() for parameter in model.parameters())
    parameter_reference = int(parity["current"]["parameters"])
    parameter_relative = abs(parameters - parameter_reference) / parameter_reference

    model.eval()
    permutation = torch.arange(args.block_count - 1, -1, -1)
    with torch.no_grad():
        original = model(target, time_value, condition)
        permuted = permuted_batch(batch, permutation)
        permuted_target, permuted_condition, _ = encode_target_and_condition(
            model,
            permuted,
        )
        reordered = model(permuted_target, time_value, permuted_condition)
    permutation_error = (original - reordered[:, permutation]).abs().max()

    empty = copy.deepcopy(batch)
    empty["b2b"].zero_()
    empty["p2b"].zero_()
    empty["n_b2b"].zero_()
    empty["n_p2b"].zero_()
    empty["n_pins"].zero_()
    empty_target, empty_condition, _ = encode_target_and_condition(model, empty)
    with torch.no_grad():
        empty_output = model(empty_target, time_value, empty_condition)

    restored = build_chip_model(model_config())
    restored.load_state_dict(copy.deepcopy(model.state_dict()))
    checkpoint_match = all(
        torch.equal(value, restored.state_dict()[key])
        for key, value in model.state_dict().items()
    )
    final_gradient = model.backbone.final.linear.weight.grad
    checks = {
        "token_prediction_shape": prediction.shape == (args.cases, args.block_count, 3),
        "finite_forward_backward": bool(
            torch.isfinite(prediction).all() and torch.isfinite(decoded).all()
        ),
        "nonzero_output_gradient": bool(
            final_gradient is not None and final_gradient.abs().sum() > 0.0
        ),
        "exact_area_decode": float(area_error) <= 1.0e-4,
        "permutation_equivariant": float(permutation_error) <= 2.0e-5,
        "empty_graph_finite": bool(torch.isfinite(empty_output).all()),
        "checkpoint_round_trip": checkpoint_match,
        "parameter_match_within_1pct": parameter_relative < 0.01,
    }
    report = {
        "status": "pass" if all(checks.values()) else "reject_or_repair",
        "scope": "CPU bridge smoke on real FloorSet cache; no quality claim",
        "cache": str(args.cache),
        "block_count": args.block_count,
        "cases": args.cases,
        "parameters": parameters,
        "parameter_reference": parameter_reference,
        "parameter_relative_gap": parameter_relative,
        "cpu_forward_backward_seconds": backward_seconds,
        "metrics": {
            "max_exact_area_error": float(area_error),
            "max_permutation_error": float(permutation_error),
            "empty_graph_max_abs": float(empty_output.abs().max()),
            "final_head_gradient_abs_sum": float(
                final_gradient.abs().sum()
            ),
        },
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
