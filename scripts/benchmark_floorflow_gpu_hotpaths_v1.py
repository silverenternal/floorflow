"""Benchmark exact-preserving residential FloorFlow hot-path rewrites."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path
from statistics import median

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from floorflow.objectives.gpu_assignment import (
    assignment_index_tensors,
    exact_assignment_batch,
)
from floorflow.objectives.token_grammar import FloorplanTokenVocabulary
from floorflow.representation.polygon_targets import collapse_invalid_polygon_points
from floorflow.training.floorplan_token_training_runtime import (
    FloorplanTokenTrainingConfig,
    FloorplanTokenTrainingRuntime,
)


def _benchmark(
    function: Callable[[], object],
    device: torch.device,
    *,
    warmup: int,
    iterations: int,
) -> float:
    for _ in range(warmup):
        function()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    samples = []
    for _ in range(iterations):
        started = time.perf_counter()
        function()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        samples.append((time.perf_counter() - started) * 1000.0)
    return median(samples)


def _record(
    left: Callable[[], object],
    right: Callable[[], object],
    device: torch.device,
    *,
    warmup: int,
    iterations: int,
) -> dict[str, float]:
    baseline = _benchmark(left, device, warmup=warmup, iterations=iterations)
    optimized = _benchmark(right, device, warmup=warmup, iterations=iterations)
    return {
        "legacy_ms": baseline,
        "optimized_ms": optimized,
        "speedup": baseline / max(optimized, 1.0e-9),
    }


def _assignment_case(
    device: torch.device,
    warmup: int,
    iterations: int,
) -> dict[str, float]:
    generator = torch.Generator(device=device).manual_seed(7)
    costs = [
        torch.rand(24, 12, device=device, generator=generator)
        for _ in range(8)
    ]

    def legacy():
        return [
            linear_sum_assignment(cost.detach().cpu().numpy())
            for cost in costs
        ]

    def optimized():
        return exact_assignment_batch(costs)

    for expected, actual in zip(legacy(), optimized()):
        assert (expected[0] == actual[0]).all()
        assert (expected[1] == actual[1]).all()
    return _record(
        legacy,
        optimized,
        device,
        warmup=warmup,
        iterations=iterations,
    )


def _wall_opening_case(
    device: torch.device,
    warmup: int,
    iterations: int,
) -> dict[str, object]:
    torch.manual_seed(3)
    batch, queries = 2, 16
    pairs = [
        (torch.arange(6).numpy(), torch.arange(6).numpy()),
        (torch.arange(11).numpy(), torch.arange(11).numpy()),
    ]
    endpoint_logits = torch.randn(
        batch,
        queries,
        2,
        2,
        device=device,
        requires_grad=True,
    )
    endpoint_targets = torch.randn(batch, queries, 2, 2, device=device)
    wall_logits = torch.randn(
        batch,
        queries,
        device=device,
        requires_grad=True,
    )
    wall_targets = torch.randint(
        0,
        2,
        (batch, queries),
        device=device,
    ).float()
    centers = torch.randn(
        batch,
        queries,
        2,
        device=device,
        requires_grad=True,
    )
    widths = torch.rand(
        batch,
        queries,
        device=device,
        requires_grad=True,
    )
    target_centers = torch.randn(batch, queries, 2, device=device)
    target_widths = torch.rand(batch, queries, device=device)
    host_logits = torch.randn(
        batch,
        queries,
        8,
        device=device,
        requires_grad=True,
    )
    host_targets = torch.randint(0, 8, (batch, queries), device=device)

    def legacy_wall():
        losses = []
        types = []
        for index, (rows, columns) in enumerate(pairs):
            row_tensor = torch.as_tensor(rows, device=device)
            column_tensor = torch.as_tensor(columns, device=device)
            predicted = endpoint_logits[index, row_tensor]
            expected = endpoint_targets[index, column_tensor]
            losses.append(
                torch.minimum(
                    F.smooth_l1_loss(
                        predicted,
                        expected,
                        reduction="none",
                    ).mean(),
                    F.smooth_l1_loss(
                        predicted,
                        expected[:, [1, 0]],
                        reduction="none",
                    ).mean(),
                )
            )
            types.append(
                F.binary_cross_entropy_with_logits(
                    wall_logits[index, row_tensor],
                    wall_targets[index, column_tensor],
                )
            )
        return torch.stack(losses).mean() + torch.stack(types).mean()

    def optimized_wall():
        rows, targets, valid = assignment_index_tensors(
            pairs,
            device=device,
        )
        batch_index = torch.arange(batch, device=device)[:, None]
        predicted = endpoint_logits[batch_index, rows]
        expected = endpoint_targets[batch_index, targets]
        direct = F.smooth_l1_loss(
            predicted,
            expected,
            reduction="none",
        ).mean(dim=(2, 3))
        reverse = F.smooth_l1_loss(
            predicted,
            expected[:, :, [1, 0]],
            reduction="none",
        ).mean(dim=(2, 3))
        endpoint_loss = torch.minimum(
            _sample_mean(direct, valid),
            _sample_mean(reverse, valid),
        )
        type_loss = _sample_mean(
            F.binary_cross_entropy_with_logits(
                wall_logits[batch_index, rows],
                wall_targets[batch_index, targets],
                reduction="none",
            ),
            valid,
        )
        return endpoint_loss + type_loss

    def legacy_opening():
        center_losses = []
        width_losses = []
        host_losses = []
        for index, (rows, columns) in enumerate(pairs):
            row_tensor = torch.as_tensor(rows, device=device)
            column_tensor = torch.as_tensor(columns, device=device)
            center_losses.append(
                F.smooth_l1_loss(
                    centers[index, row_tensor],
                    target_centers[index, column_tensor],
                )
            )
            width_losses.append(
                F.smooth_l1_loss(
                    widths[index, row_tensor],
                    target_widths[index, column_tensor],
                )
            )
            target = host_targets[index, column_tensor]
            valid = target >= 0
            losses = F.cross_entropy(
                host_logits[index, row_tensor],
                target.long().clamp_min(0),
                reduction="none",
            )
            selected = valid.to(losses.dtype)
            host_losses.append(
                (losses * selected).sum() / selected.sum().clamp_min(1.0)
            )
        return (
            torch.stack(center_losses).mean()
            + torch.stack(width_losses).mean()
            + torch.stack(host_losses).mean()
        )

    def optimized_opening():
        rows, targets, valid = assignment_index_tensors(
            pairs,
            device=device,
        )
        batch_index = torch.arange(batch, device=device)[:, None]
        center_loss = F.smooth_l1_loss(
            centers[batch_index, rows],
            target_centers[batch_index, targets],
            reduction="none",
        ).mean(dim=-1)
        width_loss = F.smooth_l1_loss(
            widths[batch_index, rows],
            target_widths[batch_index, targets],
            reduction="none",
        )
        target = host_targets[batch_index, targets]
        host_valid = target >= 0
        logits = host_logits[batch_index, rows]
        host_loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target.long().clamp_min(0).reshape(-1),
            reduction="none",
        ).reshape_as(target)
        return (
            _sample_mean(center_loss, valid)
            + _sample_mean(width_loss, valid)
            + _sample_mean(
                host_loss,
                valid & host_valid,
                include=valid.any(dim=1),
            )
        )

    assert torch.allclose(
        legacy_wall(),
        optimized_wall(),
        atol=2.0e-7,
        rtol=1.0e-6,
    )
    assert torch.allclose(
        legacy_opening(),
        optimized_opening(),
        atol=2.0e-7,
        rtol=1.0e-6,
    )
    return {
        "wall": _record(
            legacy_wall,
            optimized_wall,
            device,
            warmup=warmup,
            iterations=iterations,
        ),
        "opening": _record(
            legacy_opening,
            optimized_opening,
            device,
            warmup=warmup,
            iterations=iterations,
        ),
    }


def _sample_mean(
    values: torch.Tensor,
    valid: torch.Tensor,
    *,
    include: torch.Tensor | None = None,
) -> torch.Tensor:
    weights = valid.to(values.dtype)
    sample_sum = (values * weights).sum(dim=1)
    sample_count = weights.sum(dim=1)
    if include is None:
        include = sample_count.gt(0)
    sample_mean = sample_sum / sample_count.clamp_min(1.0)
    include = include.to(sample_mean.dtype)
    return (sample_mean * include).sum() / include.sum().clamp_min(1.0)


def _collapse_case(
    device: torch.device,
    warmup: int,
    iterations: int,
) -> dict[str, float]:
    generator = torch.Generator(device=device).manual_seed(1)
    polygons = torch.rand(
        4,
        24,
        24,
        2,
        device=device,
        generator=generator,
    )
    point_valid = (
        torch.rand(4, 24, 24, device=device, generator=generator) > 0.35
    )

    def legacy():
        output = polygons.clone()
        for batch in range(polygons.shape[0]):
            for query in range(polygons.shape[1]):
                indices = torch.nonzero(
                    point_valid[batch, query],
                    as_tuple=False,
                ).flatten()
                if indices.numel() < 3:
                    indices = torch.arange(24, device=device)
                selected = polygons[batch, query, indices]
                output[batch, query, : selected.shape[0]] = selected
                output[batch, query, selected.shape[0] :] = selected[-1]
        return output

    optimized = lambda: collapse_invalid_polygon_points(polygons, point_valid)
    assert torch.equal(legacy(), optimized())
    return _record(
        legacy,
        optimized,
        device,
        warmup=warmup,
        iterations=iterations,
    )


def _prefix_case(
    device: torch.device,
    warmup: int,
    iterations: int,
) -> dict[str, float]:
    token_ids = torch.randint(0, 32, (8, 128), device=device)
    token_mask = torch.ones_like(token_ids, dtype=torch.bool)
    positions = torch.arange(token_ids.shape[1], device=device)
    modifiable = token_mask & (positions[None] >= 5)
    vocabulary = FloorplanTokenVocabulary(
        tuple(f"T{index}" for index in range(32))
    )
    runtime = FloorplanTokenTrainingRuntime(
        FloorplanTokenTrainingConfig(
            enabled=True,
            prefix_corruption_prob=1.0,
        ),
        vocabulary=vocabulary,
    )
    runtime.step = 1000

    def legacy():
        input_ids = token_ids.clone()
        for row in range(token_ids.shape[0]):
            starts = modifiable[row].nonzero(as_tuple=False)
            start = int(starts[torch.randint(len(starts), (), device=device)])
            suffix = positions >= start
            replacements = torch.randint(
                0,
                vocabulary.model_vocab_size,
                token_ids[row].shape,
                device=device,
            )
            input_ids[row] = torch.where(
                suffix,
                replacements,
                input_ids[row],
            )
        return input_ids.masked_fill(~token_mask, -1)

    def optimized():
        return runtime._scheduled_token_inputs(token_ids, token_mask)

    assert torch.equal(optimized()[:, :5], token_ids[:, :5])
    return _record(
        legacy,
        optimized,
        device,
        warmup=warmup,
        iterations=iterations,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reports/floorflow_gpu_hotpaths_v1.json"),
    )
    args = parser.parse_args()
    if args.device.startswith("cuda") and args.device != "cuda:0":
        raise ValueError("benchmarks are restricted to cuda:0")
    device = torch.device(args.device)
    result = {
        "device": str(device),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "exact_equivalence": {
            "room_assignment": True,
            "wall_loss": True,
            "opening_loss": True,
            "polygon_point_collapse": True,
        },
        "timings": {
            "room_assignment": _assignment_case(
                device,
                args.warmup,
                args.iterations,
            ),
            **_wall_opening_case(
                device,
                args.warmup,
                args.iterations,
            ),
            "polygon_point_collapse": _collapse_case(
                device,
                args.warmup,
                args.iterations,
            ),
            "token_prefix_corruption": _prefix_case(
                device,
                args.warmup,
                args.iterations,
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
