#!/usr/bin/env python3
"""CPU probe for legalized shape-mode candidate coverage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party/Trinity/src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_chip_official import case_payload, score_case

from floorflow.chip_flow.cache import PackedCaseStore
from floorflow.chip_flow.checkpoint_loading import load_chip_checkpoint
from floorflow.chip_flow.flow_adapter import sample_for_batch
from floorflow.chip_flow.legalizer_policy import LegalizerPolicy
from floorflow.chip_flow.model import decode_boxes_batch
from floorflow.chip_flow.projections import project_conditioned_latent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT
        / "outputs/chip_hierarchy/s7_shape_mode_v1_quick/"
        "shape_mode_v1/checkpoints/final.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_shape_mode_consumer_probe_v1.json",
    )
    parser.add_argument(
        "--indices-manifest",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--selection-mode",
        choices=("learned", "pareto"),
        default="learned",
    )
    parser.add_argument(
        "--shape-mode-top-fractions",
        type=float,
        nargs="+",
        default=None,
    )
    return parser.parse_args()


def _summary(rows: list[dict], prefix: str) -> dict[str, float]:
    keys = (
        "hard_cost",
        "v_rel",
        "area_gap",
        "rho_mae",
        "hpwl_gap",
    )
    return {
        key: float(np.mean([row[f"{prefix}_{key}"] for row in rows]))
        for key in keys
    }


def main() -> None:
    args = parse_args()
    args.checkpoint = args.checkpoint.resolve()
    args.output = args.output.resolve()
    cache = ROOT / "outputs/cache/chip_floorset_50k_train_12k_dev_v2/dev"
    device = torch.device("cpu")
    store = PackedCaseStore(cache)
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    model = load_chip_checkpoint(checkpoint, "cpu", use_ema=False)
    if args.shape_mode_top_fractions is not None:
        model.shape_mode_head.mode_top_fractions = tuple(
            args.shape_mode_top_fractions
        )
    model.shape_mode_apply = False
    model.shape_mode_selection = args.selection_mode
    policy = LegalizerPolicy(
        relation="restored",
        reshape=True,
        trust=1.3,
        outline=True,
        time_limit_ms=2000,
    )
    policy.validate()
    case_rows = []
    if args.indices_manifest is not None:
        selected_indices = json.loads(
            args.indices_manifest.resolve().read_text(encoding="utf-8")
        )["indices"]
    else:
        selected_indices = []
        for count in (21, 30, 40, 50):
            indices = store.indices(count)
            if len(indices):
                selected_indices.append(int(indices[0]))
    for selected_case in selected_indices:
        count = int(store.arrays["n"][selected_case])
        selected = np.asarray([selected_case], dtype=np.int64)
        batch = store.batch(selected, device=device)
        batch["_indices"] = torch.as_tensor(selected, device=device)
        model.shape_mode_apply = False
        baseline, diagnostics = sample_for_batch(
            model,
            batch,
            sampling_steps=4,
            generator=torch.Generator(device=device).manual_seed(19),
        )
        modes = diagnostics.get("shape_mode_latents")
        if modes is None:
            raise RuntimeError("checkpoint did not produce shape-mode candidates")
        model.shape_mode_apply = True
        selected_latent, selected_diagnostics = sample_for_batch(
            model,
            batch,
            sampling_steps=4,
            generator=torch.Generator(device=device).manual_seed(19),
        )
        model.shape_mode_apply = False
        candidates = [baseline, *[modes[:, index] for index in range(modes.shape[1])]]
        rows = []
        for candidate_index, latent in enumerate(candidates):
            latent = project_conditioned_latent(
                latent,
                batch["target_latent"],
                batch["constraints"],
            )
            boxes = decode_boxes_batch(latent, batch["areas"])
            payload = case_payload(
                batch,
                0,
                boxes[0],
                0.0,
                legalize=True,
                legalizer_time_limit_ms=2000,
                legalizer_options=policy.kwargs(),
            )
            row = score_case(payload)
            row["candidate"] = candidate_index
            rows.append(row)
        selected_latent = project_conditioned_latent(
            selected_latent,
            batch["target_latent"],
            batch["constraints"],
        )
        selected_boxes = decode_boxes_batch(selected_latent, batch["areas"])
        selected_payload = case_payload(
            batch,
            0,
            selected_boxes[0],
            0.0,
            legalize=True,
            legalizer_time_limit_ms=2000,
            legalizer_options=policy.kwargs(),
        )
        selected_row = score_case(selected_payload)
        baseline_row = rows[0]
        case_rows.append(
            {
                "case": int(selected[0]),
                "blocks": count,
                "rows": rows,
                "baseline": {
                    key: baseline_row[key]
                    for key in (
                        "legalized_hard_cost",
                        "legalized_v_rel",
                        "legalized_area_gap",
                        "legalized_rho_mae",
                        "legalized_hpwl_gap",
                    )
                },
                "oracle": {
                    key: min(row[f"legalized_{key}"] for row in rows)
                    for key in (
                        "hard_cost",
                        "v_rel",
                        "area_gap",
                        "rho_mae",
                        "hpwl_gap",
                    )
                },
                "selector": {
                    "mode": int(
                        selected_diagnostics["shape_mode_selected"][0]
                    ),
                    **{
                        key: selected_row[f"legalized_{key}"]
                        for key in (
                            "hard_cost",
                            "v_rel",
                            "area_gap",
                            "rho_mae",
                            "hpwl_gap",
                        )
                    },
                },
            }
        )
    flat_rows = [row for case in case_rows for row in case["rows"]]
    report = {
        "schema_version": "chip-shape-mode-consumer-probe-v1",
        "status": "complete",
        "scope": "CPU mechanism probe; not a SOTA result",
        "checkpoint": str(args.checkpoint.relative_to(ROOT)),
        "cases": [case["case"] for case in case_rows],
        "sampling_steps": 4,
        "seed": 19,
        "selection_mode": args.selection_mode,
        "shape_mode_top_fractions": args.shape_mode_top_fractions,
        "indices_manifest": (
            str(args.indices_manifest.resolve().relative_to(ROOT))
            if args.indices_manifest is not None
            else None
        ),
        "baseline": _summary(
            [case["rows"][0] for case in case_rows],
            "legalized",
        ),
        "mode_mean": _summary(flat_rows[1:], "legalized"),
        "oracle": _summary(
            [
                {
                    f"legalized_{key}": case["oracle"][key]
                    for key in (
                        "hard_cost",
                        "v_rel",
                        "area_gap",
                        "rho_mae",
                        "hpwl_gap",
                    )
                }
                for case in case_rows
            ],
            "legalized",
        ),
        "selector": _summary(
            [
                {
                    f"legalized_{key}": case["selector"][key]
                    for key in (
                        "hard_cost",
                        "v_rel",
                        "area_gap",
                        "rho_mae",
                        "hpwl_gap",
                    )
                }
                for case in case_rows
            ],
            "legalized",
        ),
        "cases_detail": case_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output.relative_to(ROOT)),
                "baseline": report["baseline"],
                "mode_mean": report["mode_mean"],
                "oracle": report["oracle"],
                "selector": report["selector"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
