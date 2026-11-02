#!/usr/bin/env python3
"""Evaluate one routed trajectory per case under a fixed baseline budget."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party/Trinity/src"))

from trinity.sampling.refine_case import build_refine_case
from trinity.sampling.refine_closed import PAPER_WEIGHTS, refine_closed

from floorflow.chip_flow.cache import PackedCaseStore
from floorflow.chip_flow.checkpoint_loading import load_chip_checkpoint
from floorflow.chip_flow.flow_adapter import sample_for_batch
from floorflow.chip_flow.portfolio_router import (
    PortfolioRouter,
    router_features,
    validate_router_accuracy,
)
from floorflow.chip_flow.refiner_config import configure_chip_projector
from floorflow.chip_flow.reproducibility import stable_batch_seed
from floorflow.evaluation.paired_bootstrap import paired_delta
from scripts.analyze_chip_m48_displacement import load_indices
from scripts.evaluate_chip_m48_portfolio import legalized_candidate
from scripts.evaluate_chip_refiner import aggregate_physics, make_instances


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--router-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--cache",
        type=Path,
        default=ROOT / "outputs/cache/chip_official_test_100_v1",
    )
    parser.add_argument("--case-indices", type=Path, default=None)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--nfe", type=int, default=1)
    parser.add_argument("--refiner-steps", type=int, default=1600)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--minimum-router-accuracy", type=float, default=0.65)
    parser.add_argument("--allow-low-router-accuracy", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _router(path: Path, device: torch.device) -> PortfolioRouter:
    payload = torch.load(path, map_location=device, weights_only=False)
    model = PortfolioRouter(
        int(payload["config"]["hidden"]),
        int(payload["config"].get("feature_dim", 10)),
    ).to(device)
    model.load_state_dict(payload["model"])
    return model.eval()


def _reference_rows(path: Path) -> dict[int, dict]:
    report = json.loads(path.read_text(encoding="utf-8"))
    return {
        int(row["index"]): row["baseline"]["stages"]["legalized"]
        for row in report["rows"]
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = load_chip_checkpoint(checkpoint, device, use_ema=True)
    router = _router(args.router_checkpoint, device)
    accuracy = float(
        torch.load(
            args.router_checkpoint,
            map_location="cpu",
            weights_only=False,
        )["config"].get("validation_accuracy", 0.0)
    )
    validate_router_accuracy(
        accuracy,
        args.minimum_router_accuracy,
        allow_low_accuracy=args.allow_low_router_accuracy,
    )
    store = PackedCaseStore(args.cache)
    indices = (
        load_indices(args.case_indices)
        if args.case_indices
        else store.indices().tolist()
    )
    reference = _reference_rows(args.reference_report)
    rows = []
    with torch.no_grad():
        for count in sorted({int(store.arrays["n"][index]) for index in indices}):
            group = [index for index in indices if int(store.arrays["n"][index]) == count]
            for start in range(0, len(group), args.batch_size):
                case_indices = group[start : start + args.batch_size]
                batch = store.batch(np.asarray(case_indices), device=device)
                batch["_indices"] = torch.as_tensor(case_indices, device=device)
                features = router_features(
                    batch["constraints"],
                    batch["b2b"],
                    batch["p2b"],
                    batch["n_b2b"],
                    batch["n_p2b"],
                    batch["areas"],
                    batch["pins"],
                    batch["n_pins"],
                )
                profiles = router.choose(features)
                for profile in sorted({item.name for item in profiles}):
                    subset = [
                        row
                        for row, selected in enumerate(profiles)
                        if selected.name == profile
                    ]
                    selected_indices = np.asarray(
                        [case_indices[row] for row in subset],
                        dtype=np.int64,
                    )
                    subset_batch = store.batch(selected_indices, device=device)
                    subset_batch["_indices"] = torch.as_tensor(
                        selected_indices,
                        device=device,
                    )
                    choice = next(item for item in profiles if item.name == profile)
                    configured = configure_chip_projector(
                        model,
                        enabled=True,
                        iterations=choice.projector_iterations,
                        boundary_axes=False,
                        boundary_lock_corners=True,
                    )
                    generator = torch.Generator(device=device).manual_seed(
                        stable_batch_seed(args.seed, selected_indices)
                    )
                    raw, _ = sample_for_batch(
                        configured,
                        subset_batch,
                        sampling_steps=args.nfe,
                        generator=generator,
                    )
                    refine_case = build_refine_case(
                        make_instances(subset_batch),
                        1,
                        device,
                    )
                    refined = refine_closed(
                        raw,
                        refine_case,
                        PAPER_WEIGHTS,
                        args.refiner_steps,
                        1.0e-3,
                        "linear",
                        chunk=5,
                        compile_loop=False,
                        shape=True,
                        optimizer="adam",
                    )[args.refiner_steps]
                    boxes, _ = aggregate_physics(refined, subset_batch)
                    for local_row, original_row in enumerate(subset):
                        constraints = subset_batch["constraints"][local_row, :count]
                        scale = subset_batch["areas"][local_row, :count].sum().sqrt().clamp_min(1e-6)
                        candidate = legalized_candidate(
                            profile,
                            refined,
                            boxes,
                            subset_batch,
                            local_row,
                            constraints,
                            scale,
                            relation=choice.relation,
                            time_limit_ms=2000,
                            device=device,
                        )
                        index = int(case_indices[original_row])
                        rows.append(
                            {
                                "index": index,
                                "blocks": count,
                                "profile": profile,
                                "projector_iterations": choice.projector_iterations,
                                "relation": choice.relation,
                                "selected": candidate,
                            }
                        )
    metrics = {}
    for name in ("cost", "v_rel"):
        current = [
            {
                "index": row["index"],
                "value": row["selected"]["stages"]["legalized"][name],
            }
            for row in rows
            if row["index"] in reference
        ]
        baseline = [
            {"index": row["index"], "value": reference[row["index"]][name]}
            for row in rows
            if row["index"] in reference
        ]
        metrics[name] = paired_delta(
            current,
            baseline,
            "value",
            iterations=10000,
            lower_is_better=True,
        )
    report = {
        "version": "chip-router-portfolio-evaluator-v1",
        "status": "complete",
        "budget": {
            "nfe": args.nfe,
            "refiner_steps": args.refiner_steps,
            "legalizer_calls": 1,
        },
        "router_validation_accuracy": accuracy,
        "profile_counts": {
            profile: sum(row["profile"] == profile for row in rows)
            for profile in sorted({row["profile"] for row in rows})
        },
        "paired": metrics,
        "rows": rows,
        "boundary": "Development evidence until a held-out router and matched anchor run are available.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
