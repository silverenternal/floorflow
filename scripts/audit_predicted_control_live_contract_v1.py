#!/usr/bin/env python3
"""Exercise promoted predicted-control paths on one real frozen control case."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_raster_predicted_control_bridge import (
    RasterPredictedControlDataset,
)

from floorflow.data.control_cache import read_control_cache
from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.inference.predicted_control_state import (
    predicted_control_typed_state,
)
from floorflow.model.predicted_channel_checkpoint import (
    build_predicted_channel_model,
)
from floorflow.training.typed_channel_route_metrics import (
    typed_channel_route_metrics,
)

ROOT = Path(__file__).resolve().parents[1]


def _grad_present(parameter: torch.nn.Parameter) -> bool:
    return parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = payload["config"]
    taxonomy = FloorplanTaxonomy.from_json(payload["taxonomy"])
    model = build_predicted_channel_model(taxonomy, config).to(device)
    model.load_state_dict(payload["model"])
    model.train()
    dataset = RasterPredictedControlDataset(
        Path(config["predicted_root"]),
        "val",
        Path(config["cache_root"]),
        Path(config["data_root"]),
        image_size=int(config["image_size"]),
        limit=1,
    )
    _target, footprint, _floor_ids, names, _image = next(
        iter(DataLoader(dataset, batch_size=1, shuffle=False))
    )
    control_path = (
        Path(config["predicted_root"])
        / "val"
        / f"{names[0].replace('/', '__')}.npz"
    )
    controls = {
        key: value[None].to(device)
        for key, value in read_control_cache(control_path).items()
    }
    footprint = footprint.to(device)
    output = model(controls, footprint)
    state = predicted_control_typed_state(output)
    finite = all(
        bool(torch.isfinite(value).all())
        for value in output.values()
        if torch.is_tensor(value)
    )
    loss = (
        output["room_instance_logits"].square().mean()
        + 0.05 * output["stage_walls_logits"].square().mean()
        + 0.05 * output["stage_openings_logits"].square().mean()
        + 0.02 * output["furniture_query_class_logits"].square().mean()
        + 0.02 * output["furniture_query_mask_logits"].square().mean()
    )
    loss.backward()
    selection = output.get("query_count_candidate_mask")
    ghost_mask_ok = selection is None or (
        torch.isinf(state.room_exists[~selection]).all()
        or (state.room_exists[~selection] < -1.0e30).all()
    )
    checks = {
        "all_outputs_finite": finite,
        "loss_finite": bool(torch.isfinite(loss)),
        "action_refresh_3": int(output["action_refresh_count"]) == 3,
        "room_stage_context_live": _grad_present(
            model.stage_refiners["rooms"].output.weight
        ),
        "typed_channel_live": _grad_present(model.head.typed_channel_residual),
        "room_query_count_live": _grad_present(
            model.room_instance_stage.query_set_head.count[-1].weight
        ),
        "room_graph_live": _grad_present(
            model.room_instance_stage.room_graph_conditioner.scale
        ),
        "action_graph_live": _grad_present(model.action_graph.update_scale),
        "furniture_query_live": _grad_present(
            model.furniture_stage.head.decoder.class_head.weight
        ),
        "mask_class_evidence_present": (
            "mask_class_evidence" in output
            and output["mask_class_evidence"].shape[-1]
            == len(taxonomy.room_ids)
        ),
        "edit_ghost_slots_masked": bool(ghost_mask_ok),
    }
    report = {
        "schema_version": "predicted-control-live-contract-v1",
        "device": str(device),
        "case": names[0],
        "checkpoint": str(args.checkpoint),
        "checks": checks,
        "passed": all(checks.values()),
        "loss": float(loss.detach()),
        "route_metrics": typed_channel_route_metrics(output),
        "shapes": {
            "logits": list(output["logits"].shape),
            "room_instance": list(output["room_instance_logits"].shape),
            "adjacency": list(output["room_adjacency_logits"].shape),
            "furniture_class": list(
                output["furniture_query_class_logits"].shape
            ),
            "action_stage": list(output["action_stage_tokens"].shape),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
