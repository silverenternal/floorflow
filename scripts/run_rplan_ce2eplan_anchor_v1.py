"""Run CE2EPlan boundary-only inference on the shared RPLAN validation cases."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

from floorflow.data.rplan_boundary_adapter import load_rplan_rows, rplan_taxonomy
from floorflow.evaluation.ce2eplan_adapter import adapt_ce2eplan_row
from floorflow.evaluation.room_metric_suite import room_metric_suite

ROOT = Path(__file__).resolve().parents[1]
CE2E_ROOT = ROOT / ".data/external/CE2EPlan"
sys.path.insert(0, str(CE2E_ROOT))
sys.path.insert(0, str(CE2E_ROOT / "scripts/output"))

from ce2eplan.script_util import (
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    update_arg_parser,
)


def _postprocess_module():
    path = CE2E_ROOT / "scripts/output/post_processing.py"
    spec = importlib.util.spec_from_file_location("ce2e_postprocess", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load CE2EPlan post-processing")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CE2E_ROOT / "scripts/trained_model/b_model300000.pt",
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=CE2E_ROOT / "dataset/dataset_json/data_val.json",
    )
    parser.add_argument("--val-offset", type=int, default=0)
    parser.add_argument("--val-cases", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2905)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/rplan_ce2eplan_anchor_val_v1.json",
    )
    return parser.parse_args()


def _condition_tensors(rows: list[dict], device: torch.device):
    boundary = []
    door = []
    for row in rows:
        boundary_values = (
            np.asarray(row["boundary_expand"], dtype=np.float32).reshape(1, -1)
            / 256.0
            * 2.0
            - 1.0
        )
        door_values = (
            np.asarray(row["entrance_expand"], dtype=np.float32).reshape(1, -1)
            / 256.0
            * 2.0
            - 1.0
        )
        boundary.append(np.repeat(boundary_values, 8, axis=0))
        door.append(np.repeat(door_values, 8, axis=0))
    boundary_tensor = torch.from_numpy(np.stack(boundary)).to(device)
    door_tensor = torch.from_numpy(np.stack(door)).to(device)
    return {
        "syn_atten_mask": torch.zeros(
            len(rows),
            8,
            8,
            device=device,
        ),
        "syn_cond_mask": torch.ones(
            len(rows),
            6,
            8,
            device=device,
        ),
        "syn_cond_boundary": boundary_tensor,
        "syn_cond_door": door_tensor,
    }


def _row_from_tensor(values: np.ndarray, source: dict) -> dict:
    rooms = []
    for index, row in enumerate(values):
        if float(row[0]) < 0.5:
            continue
        category = round((float(row[1]) + 1.0) / 2.0 * 6.0 - 1.0)
        center_x = round((float(row[2]) + 1.0) / 2.0 * 256.0)
        center_y = round((float(row[3]) + 1.0) / 2.0 * 256.0)
        width = round((float(row[4]) + 1.0) / 2.0 * 256.0)
        height = round((float(row[5]) + 1.0) / 2.0 * 256.0)
        rooms.append(
            {
                "id": len(rooms),
                "category": min(5, max(0, category)),
                "location": [center_x, center_y],
                "size": max(0, width * height),
                "box": [
                    round(center_x - width / 2),
                    round(center_y - height / 2),
                    round(center_x + width / 2),
                    round(center_y + height / 2),
                ],
            }
        )
    return {
        "name": str(source["name"]),
        "boundary": source["boundary"],
        "boundary_expand": source["boundary_expand"],
        "entrance_expand": source["entrance_expand"],
        "rooms": rooms,
    }


def _fallback_alignment(row: dict) -> dict:
    rooms = []
    for room in row["rooms"]:
        x0, y0, x1, y1 = room["box"]
        rooms.append(
            {
                **room,
                "box_aligned": [x0, y0, x1, y1],
                "r_boundary_aligned": [
                    [x0, y0],
                    [x1, y0],
                    [x1, y1],
                    [x0, y1],
                ],
            }
        )
    return {**row, "rooms": rooms}


def main() -> None:
    args = _args()
    device = torch.device(args.device)
    if device.type == "cuda" and str(device) != "cuda:0":
        raise ValueError("CE2EPlan anchor inference is restricted to cuda:0")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model_args = argparse.Namespace(**model_and_diffusion_defaults())
    model_args.dataset = "rplan"
    model_args.support_boundary = True
    update_arg_parser(model_args)
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(model_args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    )
    model.to(device).eval()
    rows = load_rplan_rows(
        args.validation,
        offset=args.val_offset,
        limit=args.val_cases,
    )
    postprocess = _postprocess_module()
    taxonomy = rplan_taxonomy()
    outputs = []
    processed = 0
    with torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            chunk = rows[start : start + args.batch_size]
            model_kwargs = _condition_tensors(chunk, device)
            x_start = torch.zeros(
                (len(chunk), 6, 8),
                device=device,
            )
            samples = diffusion.p_sample_loop(
                model,
                x_start.shape,
                clip_denoised=True,
                model_kwargs=model_kwargs,
                x_start=x_start,
                mode="auto",
            )
            last = samples[-1].permute(0, 2, 1).cpu().numpy()
            for source, values in zip(chunk, last, strict=True):
                raw = _row_from_tensor(values, source)
                try:
                    outputs.append(postprocess.main(raw, support_boundary=True))
                except (IndexError, KeyError, RuntimeError, ValueError):
                    outputs.append(_fallback_alignment(raw))
            processed += len(chunk)
            print(json.dumps({"processed": processed}), flush=True)
    case_metrics = []
    failures = 0
    for target_row, predicted_row in zip(rows, outputs, strict=True):
        try:
            pred_labels, pred_footprint, _, _ = adapt_ce2eplan_row(
                predicted_row,
                taxonomy,
                prediction=True,
            )
            target_labels, target_footprint, _, _ = adapt_ce2eplan_row(
                target_row,
                taxonomy,
                prediction=False,
            )
            metrics = room_metric_suite(
                pred_labels,
                target_labels,
                pred_footprint.astype(bool),
                taxonomy,
                predicted_room_support=pred_footprint.astype(bool),
                target_room_support=target_footprint.astype(bool),
            )["matched_instance"]
            case_metrics.append(metrics)
        except (IndexError, KeyError, RuntimeError, ValueError):
            failures += 1
    aggregate = {
        key: float(np.mean([row[key] for row in case_metrics]))
        for key in case_metrics[0]
        if isinstance(case_metrics[0][key], (int, float))
    }
    report = {
        "schema": "rplan-ce2eplan-pretrained-anchor-v1",
        "status": "complete",
        "test_opened": False,
        "checkpoint": str(args.checkpoint),
        "checkpoint_name": args.checkpoint.name,
        "validation_cases": len(rows),
        "val_offset": args.val_offset,
        "processed_cases": len(outputs),
        "evaluation_failures": failures,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "optimizer_steps_declared_by_checkpoint": 300000,
        "training_batch_size_readme": 512,
        "training_sample_exposures_readme": 153600000,
        "sampling_steps": int(diffusion.num_timesteps),
        "metrics": aggregate,
        "claim_boundary": "Pretrained external anchor, not a matched-training comparison.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output.parent / "rplan_ce2eplan_anchor_val_v1_predictions.json").write_text(
        json.dumps(outputs, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
