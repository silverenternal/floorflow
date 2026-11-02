#!/usr/bin/env python3
"""Audit dynamic room references and mask-contact adjacency on one real case."""

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
from floorflow.model.predicted_channel_checkpoint import (
    build_predicted_channel_model,
)


def _grad_live(parameter: torch.nn.Parameter | None) -> bool:
    return (
        parameter is not None
        and parameter.grad is not None
        and bool(torch.isfinite(parameter.grad).all())
        and bool(parameter.grad.abs().sum() > 0.0)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--architecture-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    taxonomy = FloorplanTaxonomy.from_json(payload["taxonomy"])
    runtime_config = payload["config"]
    candidate = json.loads(args.architecture_config.read_text(encoding="utf-8"))
    model_config = {**runtime_config, **candidate["model"]}
    model_config.pop("estimated_parameters", None)

    baseline = build_predicted_channel_model(taxonomy, runtime_config)
    candidate_model = build_predicted_channel_model(taxonomy, model_config)
    candidate_model.train()
    dataset = RasterPredictedControlDataset(
        Path(model_config["predicted_root"]),
        "val",
        Path(model_config["cache_root"]),
        Path(model_config["data_root"]),
        image_size=int(model_config["image_size"]),
        limit=1,
    )
    _target, footprint, _floor_ids, names, _image = next(
        iter(DataLoader(dataset, batch_size=1, shuffle=False))
    )
    control_path = (
        Path(model_config["predicted_root"])
        / "val"
        / f"{names[0].replace('/', '__')}.npz"
    )
    controls = {
        key: value[None]
        for key, value in read_control_cache(control_path).items()
    }
    output = candidate_model(controls, footprint)
    finite = all(
        bool(torch.isfinite(value).all())
        for value in output.values()
        if torch.is_tensor(value)
    )
    loss = (
        output["room_instance_logits"].square().mean()
        + output["room_adjacency_logits"].square().mean()
        + output["stage_openings_logits"].square().mean()
        + 0.05 * output["stage_walls_logits"].square().mean()
    )
    loss.backward()
    dynamics = output.get("adaptive_query_diagnostics")
    room_stage = candidate_model.room_instance_stage
    checks = {
        "all_outputs_finite": finite,
        "loss_finite": bool(torch.isfinite(loss)),
        "adaptive_query_diagnostics_present": dynamics is not None,
        "adaptive_reference_head_live": _grad_live(
            room_stage.decoder.adaptive_query_refiner.reference_head[-1].weight
        ),
        "mask_contact_adjacency_present": (
            output["room_adjacency_mask_contact_logits"].shape
            == output["room_adjacency_logits"].shape
        ),
        "count_aware_graph_presence_present": (
            output["room_graph_query_presence"].shape
            == output["room_adjacency_logits"].shape[:2]
        ),
        "typed_opening_host_gate_live": _grad_live(
            candidate_model.opening_host_gate_scale
        ),
        "validation_stage_refreshed_after_furniture": (
            "action_validation_refreshed" in output
            and "validation" in output["action_tokens"]
        ),
        "contact_scale_live": _grad_live(
            room_stage.room_adjacency_head.contact_scale
        ),
        "mask_contact_gradient_live": _grad_live(
            room_stage.decoder.mask_features.weight
        ),
        "adaptive_context_gradient_live": _grad_live(
            room_stage.decoder.adaptive_query_refiner.context[-1].weight
        ),
    }
    report = {
        "schema_version": "predicted-room-dynamic-contact-audit-v1",
        "device": "cpu",
        "test_opened": False,
        "case": names[0],
        "checkpoint_taxonomy": str(args.checkpoint),
        "architecture_config": str(args.architecture_config),
        "checks": checks,
        "passed": all(checks.values()),
        "loss": float(loss.detach()),
        "parameter_count": {
            "v3": sum(parameter.numel() for parameter in baseline.parameters()),
            "candidate": sum(
                parameter.numel() for parameter in candidate_model.parameters()
            ),
        },
        "shapes": {
            "room_adjacency": list(output["room_adjacency_logits"].shape),
            "room_mask": list(output["room_instance_mask_logits"].shape),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
