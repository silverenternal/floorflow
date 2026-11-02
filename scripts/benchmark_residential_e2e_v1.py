"""Benchmark corrected completion and deterministic graph refinement."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

from floorflow.data.control_cache import read_control_cache
from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.heads.joint_control_loading import load_joint_control_repair
from floorflow.inference.room_graph_refinement import refine_room_graph
from floorflow.inference.room_repair import (
    close_room_partition,
    merge_room_fragments,
)

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "outputs/train/residential_predicted_refiner_v1/best.pt"
CONTROLS = ROOT / "outputs/control_contracts/model_predicted_full_v1/val"
OUTPUT = ROOT / "outputs/reports/residential_e2e_runtime_benchmark_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    taxonomy = FloorplanTaxonomy.read(
        ROOT / "outputs/cache/cubicasa5k_full_v2/taxonomy.json"
    )
    model, _config = load_joint_control_repair(
        args.checkpoint,
        taxonomy,
        device,
    )
    paths = sorted(CONTROLS.glob("*.npz"))[: args.batch_size]
    rows = [read_control_cache(path) for path in paths]
    room = torch.stack([row["room_logits"] for row in rows]).to(device)
    structure = torch.stack([row["structure"] for row in rows]).to(device)
    footprint = torch.ones(
        len(rows),
        1,
        room.shape[-2],
        room.shape[-1],
        device=device,
        dtype=torch.bool,
    )
    zero_room = torch.zeros_like(room[:, :1])
    zero_structure = torch.zeros_like(structure[:, :1])

    def forward():
        return model(
            room,
            structure,
            footprint,
            zero_room,
            zero_structure,
            program=torch.stack([row["program"] for row in rows]).to(device),
        )

    with torch.inference_mode():
        for _ in range(args.warmup):
            output = forward()
        torch.cuda.synchronize(device)
        started = torch.cuda.Event(enable_timing=True)
        ended = torch.cuda.Event(enable_timing=True)
        started.record()
        for _ in range(args.iterations):
            output = forward()
        ended.record()
        torch.cuda.synchronize(device)
    latency_ms = started.elapsed_time(ended) / args.iterations

    room_ids = torch.tensor(taxonomy.room_ids, device=device)
    labels = room_ids[output["room_logits_full"].argmax(dim=1)]
    labels = F.interpolate(
        labels[:, None].float(),
        size=room.shape[-2:],
        mode="nearest",
    )[:, 0].long()
    label_np = labels[0].cpu().numpy()
    footprint_np = np.ones_like(label_np, dtype=bool)
    graph_started = time.perf_counter()
    for _ in range(args.iterations):
        smoothed = gaussian_filter(
            output["room_logits_full"][0].float().cpu().numpy(),
            sigma=(0.0, 10.0, 10.0),
        )
        decoded = np.asarray(taxonomy.room_ids)[smoothed.argmax(axis=0)]
        decoded = np.where(footprint_np, decoded, 0).astype(np.int64)
        decoded = merge_room_fragments(decoded, taxonomy, min_area=32)
        decoded = close_room_partition(decoded, footprint_np, taxonomy)
        refine_room_graph(
            decoded,
            taxonomy,
            footprint=footprint_np,
            method="direct_tiled",
        )
    graph_latency_ms = (
        (time.perf_counter() - graph_started) * 1000.0
    ) / args.iterations
    report = {
        "schema_version": "residential-e2e-runtime-benchmark-v1",
        "status": "complete",
        "device": str(device),
        "batch_size": len(rows),
        "iterations": args.iterations,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "refiner_latency_ms_per_batch": latency_ms,
        "refiner_latency_ms_per_case": latency_ms / len(rows),
        "graph_latency_ms_per_case": graph_latency_ms,
        "extra_neural_nfe": 1,
        "graph_nfe": 0,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "decode": {
            "room_logit_gaussian_sigma": 10.0,
            "merge_fragment_min_area": 32,
            "graph_method": "direct_tiled",
        },
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
