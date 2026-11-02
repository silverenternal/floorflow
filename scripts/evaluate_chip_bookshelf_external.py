#!/usr/bin/env python3
"""Evaluate FloorFlow checkpoints on the frozen GSRC/MCNC structural cohort."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party/Trinity/src"))

from trinity.floorplan.data import with_fixed_outline
from trinity.floorplan.data.gsrc import load_gsrc_case
from trinity.floorplan.data.mcnc import load_mcnc_case
from trinity.floorplan.legalize import apply_route
from trinity.floorplan.scoring.bookshelf import clamp_aspect, orient_targets, score_bookshelf
from trinity.floorplan.types import Placement
from trinity.sampling.refine_case import build_refine_case
from trinity.sampling.refine_closed import PAPER_WEIGHTS, refine_closed

from floorflow.chip_flow.checkpoint_loading import load_chip_checkpoint
from floorflow.chip_flow.flow_adapter import sample_for_batch
from floorflow.chip_flow.hierarchical_flow import outline_condition_from_hw
from floorflow.chip_flow.model import block_features, decode_boxes_batch
from floorflow.chip_flow.outline_packing import pack_fixed_outline
from floorflow.chip_flow.ppa_external import shelf_pack_fixed
from floorflow.chip_flow.projections import project_conditioned_latent
from floorflow.chip_flow.proposals import prepare_flow_pair
from floorflow.chip_flow.refiner_config import parse_refiner_weight_overrides

ROUTE = [{"name": "scale_pack", "relation": "restored", "reshape": True, "trust": 1.3}]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "outputs/reports/chip_v46_bookshelf_holdout_manifest.json",
    )
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--nfe", type=int, default=4)
    parser.add_argument("--refiner-steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--draws", type=int, default=1)
    parser.add_argument("--refiner-weight", action="append", default=[])
    parser.add_argument(
        "--outline-shelf-fallback",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Try the existing deterministic shelf pack when scale-pack misses the fixed outline.",
    )
    parser.add_argument(
        "--outline-milp-time-limit-ms",
        type=int,
        default=2000,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_v46_bookshelf_external_eval.json",
    )
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def case_from_id(case_id: str, data_root: Path):
    suite, variant, name = case_id.split("/")
    if suite == "gsrc":
        return load_gsrc_case(
            int(name[1:]),
            variant.upper(),
            root=data_root,
            allow_download=False,
        )
    return load_mcnc_case(
        name,
        variant.upper(),
        root=data_root,
        allow_download=False,
    )


def batch_from_instance(instance, device: torch.device) -> tuple[dict, object]:
    outlined = with_fixed_outline(instance, gamma=0.10, aspect=1.0, map_pins=False)
    areas = torch.as_tensor(instance.area_targets, dtype=torch.float32, device=device)
    if instance.gt_positions is not None:
        boxes = torch.as_tensor(instance.gt_positions, dtype=torch.float32, device=device)
    else:
        side = areas.sqrt()
        boxes = torch.stack(
            [
                torch.zeros_like(side),
                torch.zeros_like(side),
                side,
                side,
            ],
            dim=1,
        )
    constraints = torch.as_tensor(instance.constraints, dtype=torch.float32, device=device)
    b2b = torch.as_tensor(instance.b2b, dtype=torch.float32, device=device)
    p2b = torch.as_tensor(instance.p2b, dtype=torch.float32, device=device)
    pins = torch.as_tensor(instance.pins_pos, dtype=torch.float32, device=device)
    source, target = prepare_flow_pair(boxes.cpu(), areas.cpu())
    batch = {
        "n": instance.block_count,
        "areas": areas[None],
        "boxes": boxes[None],
        "constraints": constraints[None],
        "features": block_features(areas, constraints, pins, b2b)[None],
        "source_latent": source[None].to(device),
        "target_latent": target[None].to(device),
        "b2b": b2b[None],
        "p2b": p2b[None],
        "pins": pins[None],
        "n_b2b": torch.tensor([len(b2b)], device=device),
        "n_p2b": torch.tensor([len(p2b)], device=device),
        "n_pins": torch.tensor([len(pins)], device=device),
        "outline_state": outline_condition_from_hw(
            torch.tensor([outlined.outline], dtype=areas.dtype),
            areas[None],
        ),
    }
    return batch, outlined


@torch.no_grad()
def evaluate_case(
    model,
    instance,
    args: argparse.Namespace,
    draw_seed: int,
) -> dict:
    device = torch.device(args.device)
    batch, outlined = batch_from_instance(instance, device)
    generator = torch.Generator(device=device).manual_seed(draw_seed)
    latent, _ = sample_for_batch(
        model,
        batch,
        sampling_steps=args.nfe,
        generator=generator,
    )
    base_weights = dict(PAPER_WEIGHTS)
    base_weights["outline"] = 1.0
    weights = parse_refiner_weight_overrides(args.refiner_weight, base=base_weights)
    refine_case = build_refine_case([outlined], 1, device)
    refined = refine_closed(
        latent,
        refine_case,
        weights,
        args.refiner_steps,
        1.0e-3,
        "linear",
        chunk=5,
        optimizer="adam",
    )[args.refiner_steps]
    refined = project_conditioned_latent(
        refined,
        batch["target_latent"],
        batch["constraints"],
    )
    boxes = decode_boxes_batch(refined, batch["areas"])[0].cpu().numpy()
    legalized = apply_route(Placement(boxes, outlined), ROUTE)
    if args.outline_shelf_fallback:
        legalized = Placement(
            clamp_aspect(legalized.instance, legalized.xywh),
            legalized.instance,
        )
    legalized = Placement(
        legalized.xywh,
        orient_targets(outlined, legalized.xywh),
    )
    score = score_bookshelf(legalized)
    route_name = "scale_pack"
    if args.outline_shelf_fallback and score.fits_outline is False:
        fallback = shelf_pack_fixed(
            legalized.xywh,
            origin=(0.0, 0.0),
            outline=outlined.outline,
            position_aware_ties=True,
        )
        fallback_route = "scale_pack_outline_shelf_fallback"
        if fallback is None:
            milp = pack_fixed_outline(
                legalized.xywh,
                outlined.outline,
                time_limit_ms=args.outline_milp_time_limit_ms,
                rotatable=np.ones(outlined.block_count, dtype=bool),
            )
            fallback = None if milp is None else milp.xywh
            fallback_route = "scale_pack_outline_milp"
        if fallback is not None:
            fallback = clamp_aspect(outlined, fallback)
            fallback_placement = Placement(
                fallback,
                orient_targets(outlined, fallback),
            )
            fallback_score = score_bookshelf(fallback_placement)
            if fallback_score.fits_outline is not False:
                legalized = fallback_placement
                score = fallback_score
                route_name = fallback_route
    return {
        "legalizer_route": route_name,
        "feasible": bool(
            score.overlap_free
            and score.shapes_kept
            and score.fits_outline is not False
        ),
        "hpwl": score.hpwl,
        "bbox_area": score.area,
        "width": score.width,
        "height": score.height,
        "dead_space": score.dead_space,
        "fits_outline": score.fits_outline,
        "overlap_free": score.overlap_free,
        "shapes_kept": score.shapes_kept,
    }


def summary(rows: list[dict], arm: str) -> dict:
    feasible = [row for row in rows if row[arm]["selected"]["feasible"]]
    return {
        "cases": len(rows),
        "feasible_cases": len(feasible),
        "feasible_fraction": len(feasible) / max(1, len(rows)),
        "mean_hpwl_all": float(
            np.mean([row[arm]["selected"]["hpwl"] for row in rows])
        ),
        "mean_hpwl_feasible": (
            float(
                np.mean([row[arm]["selected"]["hpwl"] for row in feasible])
            )
            if feasible
            else None
        ),
        "mean_dead_space": float(
            np.mean([row[arm]["selected"]["dead_space"] for row in feasible])
        )
        if feasible
        else None,
    }


def main() -> None:
    args = parse_args()
    args.candidate = args.candidate.resolve()
    args.control = args.control.resolve()
    args.manifest = args.manifest.resolve()
    args.output = args.output.resolve()
    manifest = load(args.manifest)
    selected = args.case_id or [case["id"] for case in manifest["cases"]]
    available = {case["id"] for case in manifest["cases"]}
    if not set(selected).issubset(available):
        raise ValueError("requested a case outside the frozen manifest")
    device = torch.device(args.device)
    models = {
        "candidate": load_chip_checkpoint(
            torch.load(args.candidate, map_location=device, weights_only=False),
            args.device,
            use_ema=True,
        ),
        "control": load_chip_checkpoint(
            torch.load(args.control, map_location=device, weights_only=False),
            args.device,
            use_ema=True,
        ),
    }
    rows = []
    started = time.perf_counter()
    for case_id in selected:
        instance = case_from_id(case_id, ROOT / ".data")
        row = {"id": case_id, "block_count": instance.block_count}
        for arm, model in models.items():
            draws = [
                evaluate_case(
                    model,
                    instance,
                    args,
                    args.seed + 1000 * draw,
                )
                for draw in range(args.draws)
            ]
            feasible = [draw for draw in draws if draw["feasible"]]
            selected_draw = min(feasible or draws, key=lambda draw: draw["hpwl"])
            row[arm] = {
                "selected": selected_draw,
                "feasible_draws": len(feasible),
                "draws": draws,
            }
        row["hpwl_relative"] = (
            row["candidate"]["selected"]["hpwl"]
            / row["control"]["selected"]["hpwl"]
            - 1.0
        )
        rows.append(row)
        print(case_id, row["hpwl_relative"], flush=True)
    paired_rows = [
        row
        for row in rows
        if row["candidate"]["selected"]["feasible"]
        and row["control"]["selected"]["feasible"]
    ]
    mean_hpwl_relative = (
        float(np.mean([row["hpwl_relative"] for row in paired_rows]))
        if paired_rows
        else None
    )
    report = {
        "version": "chip-v46-bookshelf-external-eval-v1",
        "status": "complete" if len(rows) == len(selected) else "incomplete",
        "scope": "External structural generalization; not ORFS routed-PPA evidence.",
        "manifest": str(args.manifest.relative_to(ROOT)),
        "candidate": str(args.candidate.relative_to(ROOT)),
        "control": str(args.control.relative_to(ROOT)),
        "device": args.device,
        "nfe": args.nfe,
        "refiner_steps": args.refiner_steps,
        "seed": args.seed,
        "draws": args.draws,
        "refiner_weight_overrides": list(args.refiner_weight),
        "cases": rows,
        "summary": {
            "candidate": summary(rows, "candidate"),
            "control": summary(rows, "control"),
            "paired_feasible_cases": len(paired_rows),
            "mean_hpwl_relative_feasible_pairs": mean_hpwl_relative,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "cases": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
