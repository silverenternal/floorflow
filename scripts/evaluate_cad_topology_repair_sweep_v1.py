#!/usr/bin/env python3
"""Sweep repair keep thresholds and bounded edit budgets on validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_cad_topology_repair_v1 import candidate_inputs
from train_raster_predicted_control_bridge import RasterPredictedControlDataset
from train_residential_predicted_refiner import _predicted_controls

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.cad_candidate_plan import candidate_plan_metrics
from floorflow.evaluation.paired_bootstrap import paired_delta
from floorflow.heads.cad_topology_repair import (
    CadTopologyRepairHead,
    apply_bounded_topology_repair,
)
from floorflow.inference.cad_stage_runtime import CadStageRuntime
from floorflow.model.predicted_channel_checkpoint import (
    build_predicted_channel_model,
)
from floorflow.training.cad_wall_candidate_training import build_candidate_batch

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--layer-checkpoint",
        type=Path,
        default=ROOT / "outputs/train/cad_furniture_dense_fast64_v1/best.pt",
    )
    parser.add_argument(
        "--repair-checkpoint",
        type=Path,
        default=ROOT / "outputs/train/cad_topology_repair_fast64_v1/best.pt",
    )
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-walls", type=int, default=128)
    parser.add_argument("--minimum-room-area", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "outputs/reports/cad_topology_repair_fast64_sweep_v1.json",
    )
    return parser.parse_args()


def aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    layer_payload = torch.load(
        args.layer_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    repair_payload = torch.load(
        args.repair_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    config = layer_payload["config"]
    taxonomy = FloorplanTaxonomy.from_json(layer_payload["taxonomy"])
    layer = build_predicted_channel_model(taxonomy, config).to(device)
    layer.load_state_dict(layer_payload["model"])
    layer.requires_grad_(False).eval()
    repair = CadTopologyRepairHead(
        candidate_dim=8,
        validation_dim=int(config["field_channels"]),
        hidden_dim=int(repair_payload["config"]["hidden_dim"]),
        heads=4,
    ).to(device)
    repair.load_state_dict(repair_payload["head"])
    repair.eval()
    stage_runtime = CadStageRuntime(
        taxonomy,
        device,
        room_checkpoint=config["room_checkpoint"],
        structure_checkpoint=config["structure_checkpoint"],
        centerline_checkpoint=config["centerline_checkpoint"],
        opening_checkpoint=config["opening_checkpoint"],
        room_blend=float(config["room_blend"]),
        structure_size=128,
    )
    dataset = RasterPredictedControlDataset(
        Path(config["predicted_root"]),
        "val",
        Path(config["cache_root"]),
        Path(config["data_root"]),
        image_size=int(config["image_size"]),
        limit=args.limit,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    arms = {
        (keep_threshold, max_edits): []
        for keep_threshold in (0.2, 0.5)
        for max_edits in (2, 8)
    }
    edit_rows = {key: [] for key in arms}
    control_rows = []
    case_rows = {"control": []}
    case_rows.update({key: [] for key in arms})
    for target, footprint, floor_ids, names, image in loader:
        target = target.to(device)
        footprint = footprint.to(device)
        controls = _predicted_controls(
            Path(config["predicted_root"]),
            "val",
            names,
            args,
            device,
        )
        controls, _ = stage_runtime(
            controls,
            footprint,
            image.to(device),
            floor_ids.to(device),
        )
        output = layer(controls, footprint)
        candidates = build_candidate_batch(
            controls,
            output,
            footprint,
            taxonomy,
            minimum_room_area=args.minimum_room_area,
            max_walls=args.max_walls,
            device=device,
        )
        inputs = candidate_inputs(output, candidates)
        prediction = repair(**inputs)
        opening = F.interpolate(
            controls["opening_family_probability"].float(),
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        for index in range(target.shape[0]):
            common = {
                "endpoints": candidates.endpoints[index : index + 1],
                "scores": candidates.evidence[index : index + 1],
                "valid": candidates.valid[index : index + 1],
                "rooms": [candidates.rooms[index]],
                "host_walls": [candidates.host_walls[index]],
                "opening_probability": opening[index : index + 1],
                "target": target[index : index + 1],
                "taxonomy": taxonomy,
                "threshold": 0.2,
            }
            control_row = candidate_plan_metrics(**common)
            control_rows.append(control_row)
            case_rows["control"].append({"index": index, **control_row})
            for (keep_threshold, max_edits), rows in arms.items():
                repair_row = candidate_plan_metrics(
                    **common,
                    repair_keep_logits=prediction["wall_keep_logits"][
                        index : index + 1
                    ],
                    repair_endpoint_delta=prediction["endpoint_delta"][
                        index : index + 1
                    ],
                    repair_max_edits=max_edits,
                    repair_keep_threshold=keep_threshold,
                )
                rows.append(repair_row)
                case_rows[(keep_threshold, max_edits)].append(
                    {"index": index, **repair_row}
                )
                bounded = apply_bounded_topology_repair(
                    candidates.endpoints[index : index + 1],
                    candidates.valid[index : index + 1],
                    prediction["wall_keep_logits"][index : index + 1],
                    prediction["endpoint_delta"][index : index + 1],
                    keep_threshold=keep_threshold,
                    max_edits=max_edits,
                )
                edit_rows[(keep_threshold, max_edits)].append(
                    {
                        "edit_rate": float(
                            bounded["edited"].sum()
                            / candidates.valid[index].sum().clamp_min(1)
                        ),
                        "deleted_walls": float(bounded["removed"].sum()),
                        "moved_endpoints": float(bounded["moved"].sum()),
                        "max_edits_per_case": float(
                            bounded["edited"].sum(dim=1).max()
                        ),
                    }
                )
    control = aggregate(control_rows)
    report = {
        "schema_version": "cad-topology-repair-sweep-v1",
        "test_opened": False,
        "control": control,
        "arms": {
            f"keep_{keep}_edits_{edits}": {
                "metrics": aggregate(rows),
                "delta": {
                    key: aggregate(rows)[key] - control[key]
                    for key in control
                },
                "edits": aggregate(edit_rows[(keep, edits)]),
                "paired_wall": paired_delta(
                    case_rows[(keep, edits)],
                    case_rows["control"],
                    "wall",
                ),
                "paired_opening": paired_delta(
                    case_rows[(keep, edits)],
                    case_rows["control"],
                    "opening",
                ),
            }
            for (keep, edits), rows in arms.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
